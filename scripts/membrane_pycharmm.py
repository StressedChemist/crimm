"""Runnable example: build, export, and minimize a solvated membrane.

Dry runs omit water and ions. They do not establish solvated-membrane readiness.
The helix is a capped 1gzm fragment, not a complete rhodopsin simulation model.
"""

import argparse
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crimm.IO import PDBParser, write_crd, write_psf
from crimm.IO.CRDParser import CRDParser
from crimm.IO.PSFReader import read_psf
from crimm.IO.PSFWriter import validate_psf
from crimm.Modeller import MembraneOrienter, TopologyGenerator
from crimm.Modeller.MembraneBuilder import MembraneBuilder, MembraneSpec
from crimm.Modeller.MembraneOrienter import PPM3_MEMBRANE_TYPES
from crimm.Modeller.MembranePresets import get_membrane_preset, list_membrane_presets
from crimm.Modeller.Solvator import PERIODIC_WATER_OO_CUTOFF, wrap_periodic
from crimm.charmm_gui_9qsy.pentest import check_ring_penetration
from crimm.charmm_gui_9qsy.checkfft import checkfft
import networkx as nx

from crimm.StructEntities.Model import Model
from crimm.StructEntities.OrganizedModel import OrganizedModel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_record(directory, record, stage):
    record["stage"] = stage
    (directory / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def coordinate_pair(directory, crd_name="system.crd"):
    psf = read_psf(str(directory / "system.psf"))
    entries = CRDParser(QUIET=True).create_namedtuples(str(directory / crd_name))
    expected = [(a.segid, a.resid, a.resname, a.atomname) for a in psf.atoms]
    actual = [(a.segid, str(a.resid), a.resname, a.atomname) for a in entries]
    if expected != actual:
        raise ValueError("The PSF and CRD atom identities differ in sequence.")
    coords = np.array([a.coord for a in entries])
    if not np.isfinite(coords).all():
        raise ValueError("The CRD contains nonfinite coordinates.")
    return psf, entries, coords


CHIRAL_CENTRES = {
    "CHL1": {"C3": ("O3", "C2", "C4"), "C8": ("C7", "C9", "C14"),
             "C9": ("C8", "C10", "C11"), "C10": ("C1", "C5", "C9"),
             "C13": ("C12", "C14", "C17"), "C14": ("C8", "C13", "C15"),
             "C17": ("C13", "C16", "C20"), "C20": ("C17", "C21", "C22")},
}
CIS_BONDS = {"POPC": (("C28", "C29", "C210", "C211"),),
             "DOPC": (("C28", "C29", "C210", "C211"), ("C38", "C39", "C310", "C311"))}
STEROL_CHIRAL_SIGNS = {"C3": -1, "C8": 1, "C9": -1, "C10": -1,
                      "C13": -1, "C14": -1, "C17": 1, "C20": 1}


def dihedral_angle(points):
    a, b, c, d = np.asarray(points)
    axis = (c - b) / np.linalg.norm(c - b)
    first = a - b - np.dot(a - b, axis) * axis
    last = d - c - np.dot(d - c, axis) * axis
    return float(np.degrees(np.arctan2(np.dot(np.cross(axis, first), last), np.dot(first, last))))


def geometry_checks(model, library, cell, membrane_core_z=15.0):
    def volume(coords, centre, neighbours):
        return float(np.linalg.det([coords[name] - coords[centre] for name in neighbours]))
    reversed_lipids, trans_lipids, inward_heads = [], [], []
    leaflet_counts = [0, 0]
    leaflet_composition = [{}, {}]
    waters_in_core = ions_in_core = 0
    water_oxygens, solute_coords = [], []
    atoms, residue_ids, membrane_atoms, heavy_atoms = [], [], [], []
    graph, graph_coords, atom_ids = nx.Graph(), {}, {}
    for number, residue in enumerate(model.get_residues()):
        membrane = residue.parent.chain_type in {"Lipid", "Sterol"}
        if membrane:
            coords = {atom.name: atom.coord for atom in residue}
            template = library.templates.get(residue.resname)
            reference_index = template["valid_conformers"][0] if template is not None else 0
            reference = dict(zip(library._atoms_by_resname[residue.resname],
                                 library._coords_by_resname[residue.resname][reference_index]))
            centres = (template["chiral_centres"] if template is not None else
                       CHIRAL_CENTRES.get(residue.resname, {"C2": ("C1", "C3", "O21")}))
            for centre, neighbours in centres.items():
                if np.sign(volume(coords, centre, neighbours)) != np.sign(volume(reference, centre, neighbours)):
                    reversed_lipids.append([residue.segid, residue.id[1], residue.resname, centre])
            chain_type = residue.parent.chain_type
            head = np.mean([atom.coord for atom in MembraneBuilder._head_atoms(residue, chain_type)], axis=0)
            side = 1 if head[2] >= 0 else -1
            leaf = 0 if side == 1 else 1
            leaflet_counts[leaf] += 1
            leaflet_composition[leaf][residue.resname] = leaflet_composition[leaf].get(residue.resname, 0) + 1
            tail = np.mean([atom.coord[2] for atom in MembraneBuilder._tail_atoms(residue, chain_type)])
            if side * (head[2] - tail) <= 0:
                inward_heads.append([residue.segid, residue.id[1], residue.resname])
            for guard in template["dihedral_guards"] if template is not None else ():
                if guard["kind"] != "double_bond":
                    continue
                actual = dihedral_angle([coords[name] for name in guard["names"]])
                if abs((actual - guard["angle"] + 180) % 360 - 180) > 90:
                    trans_lipids.append([residue.segid, residue.id[1], residue.resname, *guard["names"][1:3]])
            for a, b, c, d in (() if template is not None else CIS_BONDS.get(residue.resname, ())):
                bond = coords[c] - coords[b]
                bond /= np.linalg.norm(bond)
                first, second = coords[a] - coords[b], coords[d] - coords[c]
                first -= bond * np.dot(first, bond)
                second -= bond * np.dot(second, bond)
                if np.dot(first, second) <= 0:
                    trans_lipids.append([residue.segid, residue.id[1], residue.resname, b, c])
        if residue.resname == "TIP3":
            if set(residue.child_dict) != {"OH2", "H1", "H2"}:
                raise ValueError("A water molecule is incomplete.")
            waters_in_core += any(abs(float(atom.coord[2])) < membrane_core_z for atom in residue)
            water_oxygens.append(residue["OH2"].coord)
        elif residue.parent.chain_type == "Ion":
            ions_in_core += any(abs(float(atom.coord[2])) < membrane_core_z for atom in residue)
        else:
            solute_coords.extend(atom.coord for atom in residue)
        for atom in residue:
            node = len(atoms)
            atom_ids[id(atom)] = node
            atoms.append(atom.coord)
            residue_ids.append(number)
            membrane_atoms.append(membrane)
            heavy = atom.element not in {"H", "D"}
            heavy_atoms.append(heavy)
            if heavy and residue.resname not in {"TIP3", "POT", "CLA"}:
                graph.add_node(node, segid=residue.segid, resid=str(residue.id[1]),
                               resname=residue.resname, name=atom.name)
                graph_coords[node] = atom.coord
    for chain in model:
        if chain.topology is not None:
            for first, second in chain.topology.bonds:
                a, b = atom_ids[id(first)], atom_ids[id(second)]
                if a in graph and b in graph:
                    graph.add_edge(a, b)
    penetrations, rings = check_ring_penetration(graph, graph_coords, pbc=list(cell[:2])) if graph else ([], [])
    wrapped = wrap_periodic(np.array(atoms) + np.array(cell) / 2, cell)
    pairs = KDTree(wrapped, boxsize=cell).query_pairs(1.2)
    distances = []
    heavy_contacts = 0
    for first, second in pairs:
        if (residue_ids[first] != residue_ids[second]
                and (membrane_atoms[first] or membrane_atoms[second]
                     or (heavy_atoms[first] and heavy_atoms[second]))):
            delta = wrapped[first] - wrapped[second]
            delta -= np.array(cell) * np.rint(delta / cell)
            distances.append(float(np.linalg.norm(delta)))
            heavy_contacts += heavy_atoms[first] and heavy_atoms[second]
    minimum_water_solute = minimum_water_oxygen = None
    if water_oxygens:
        oxygens = wrap_periodic(water_oxygens, cell)
        if solute_coords:
            minimum_water_solute = float(KDTree(wrap_periodic(solute_coords, cell), boxsize=cell).query(oxygens)[0].min())
        if len(oxygens) > 1:
            minimum_water_oxygen = float(KDTree(oxygens, boxsize=cell).query(oxygens, k=2)[0][:, 1].min())
    return {
        "reversed_lipids": reversed_lipids,
        "trans_lipids": trans_lipids,
        "inward_lipid_heads": inward_heads,
        "actual_leaflet_counts": leaflet_counts,
        "actual_leaflet_composition": leaflet_composition,
        "ring_penetrations": [[graph.nodes[a], graph.nodes[b]] for a, b in penetrations],
        "waters_in_core": waters_in_core,
        "ions_in_core": ions_in_core,
        "minimum_water_oxygen_to_solute_A": minimum_water_solute,
        "minimum_water_oxygen_pair_A": minimum_water_oxygen,
        "periodic_water_face_gaps_A": (np.array(cell) - np.ptp(np.mod(
            np.array(water_oxygens) + np.array(cell) / 2, cell), axis=0)).tolist() if water_oxygens else None,
        "periodic_contacts_below_1_2_A": len(distances),
        "periodic_heavy_contacts_below_1_2_A": int(heavy_contacts),
        "minimum_contact_A": min(distances) if distances else None,
    }


def protein_checks(directory, segments, orientation, crd_name="minimized.crd"):
    data, _, first = coordinate_pair(directory)
    _, _, last = coordinate_pair(directory, crd_name)
    backbone = [i for i, atom in enumerate(data.atoms) if atom.segid in segments and atom.atomname in {"N", "CA", "C"}]
    alpha = [i for i, atom in enumerate(data.atoms) if atom.segid in segments and atom.atomname == "CA"]
    fit = SVDSuperimposer()
    fit.set(first[backbone], last[backbone])
    fit.run()
    def tilt(coords):
        axis = coords[alpha[-1]] - coords[alpha[0]]
        return float(np.degrees(np.arccos(np.clip(abs(axis[2]) / np.linalg.norm(axis), 0, 1))))
    result = {"backbone_rmsd_A": float(fit.get_rms()),
              "ca_mean_z_change_A": float(last[alpha, 2].mean() - first[alpha, 2].mean()),
              "ca_endpoint_axis_tilt_change_deg": tilt(last) - tilt(first)}
    nitrogen = next(i for i, atom in enumerate(data.atoms) if atom.segid in segments and atom.atomname == "N")
    if int(data.atoms[nitrogen].resid) != orientation["topology_reference"]["residue"][1]:
        raise ValueError("The protein topology reference changed during export.")
    result["first_backbone_n_z_A"] = float(last[nitrogen, 2])
    result["topology_preserved"] = result["first_backbone_n_z_A"] * (1 if orientation["topology"] == "out" else -1) > 0.002
    result["passed"] = (result["backbone_rmsd_A"] <= 1.0
        and abs(result["ca_mean_z_change_A"]) <= 2.0
        and abs(result["ca_endpoint_axis_tilt_change_deg"]) <= 5.0 and result["topology_preserved"])
    return result


def prepare(directory, args, record):
    np.random.seed(args.seed)
    model = Model(0)
    topo = TopologyGenerator()
    nvt_steps, npt_steps = (getattr(args, name, 0) for name in ("nvt_steps", "npt_steps"))
    if nvt_steps < 0 or npt_steps < 0:
        raise ValueError("Dynamics step counts must be nonnegative.")
    if getattr(args, "short_range_warmup", False):
        if args.dry:
            raise ValueError("The short-range warmup requires a solvated membrane.")
        record["short_range_warmup"] = True
    if nvt_steps or npt_steps:
        if args.dry:
            raise ValueError("The NVT/NPT trial requires a solvated membrane.")
        if find_spec("MDAnalysis") is None:
            raise ImportError("The NVT/NPT trial requires MDAnalysis for trajectory checks.")
        record["relaxation"] = {"nvt_steps": nvt_steps, "npt_steps": npt_steps,
                                "timestep_ps": 0.001, "equilibrated": False}
    preset_name = getattr(args, "membrane", None)
    if preset_name is None and getattr(args, "cholesterol_percent", None) is not None:
        raise ValueError("--cholesterol-percent requires a PC/cholesterol --membrane preset.")
    preset = get_membrane_preset(preset_name, getattr(args, "cholesterol_percent", None)) if preset_name else None
    if preset and (args.upper_lipid_ratios is not None or args.lower_lipid_ratios is not None):
        raise ValueError("Named presets cannot be combined with custom leaflet ratios.")
    ppm_membrane = args.ppm_membrane or (preset["ppm_membrane"] if preset else "OPC")
    temperature = getattr(args, "temperature", None)
    if temperature is None:
        temperature = preset.get("temperature_K") if preset else 293.0
    if temperature is None:
        raise ValueError("PC/cholesterol families require an explicit positive temperature in K.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive in K.")
    record["temperature_K"] = temperature
    if args.case == "helix":
        archive_path = ROOT / "tests/data/ppm2_reference.tar.gz"
        with tarfile.open(archive_path) as archive:
            member = next(m for m in archive.getmembers() if Path(m.name).name == "1gzm.pdb")
            pdb_path = directory / "helix_source.pdb"
            pdb_path.write_bytes(archive.extractfile(member).read())
        model = PDBParser(QUIET=True).get_structure(str(pdb_path))[0]
        for chain in list(model):
            if chain.chain_type != "Polypeptide(L)":
                model.detach_child(chain.id)
                continue
            for residue in list(chain):
                if not 35 <= residue.id[1] <= 65:
                    chain.detach_child(residue.id)
                elif residue.resname == "HIS":
                    residue.resname = "HSE"
        record["protein_model"] = "1gzm residues 35-65, ACE/CT3, explicitly selected neutral HSE65"
        record["protein_source_sha256"] = digest(pdb_path)
    model = OrganizedModel(model, fetch_web_data=False)
    topo.generate_model(model, prot_first_patch="ACE", prot_last_patch="CT3", build_coords=True, QUIET=True)
    if args.case == "helix":
        result = MembraneOrienter(model, ph=7.5, temperature=temperature, topology=args.topology, membrane_type=ppm_membrane).orient()
        record["orientation"] = {"native_version": result.native_version, "source_revision": result.source_revision,
                                  "membrane_type": result.membrane_type, "topology": result.topology,
                                  "topology_reference": result.topology_reference,
                                  "equilibrium_thickness_A": result.equilibrium_thickness,
                                  "matching_thickness_A": result.matching_thickness,
                                  "mismatch_coefficient": result.mismatch_coefficient,
                                  "profile": result.profile, "energy": result.score,
                                  "ph": result.ph, "temperature_K": result.temperature,
                                  "thickness_A": result.membrane_thickness,
                                  "rotation": result.rotation.tolist(), "translation": result.translation.tolist()}
        native_path = find_spec("crimm.Modeller._ppm3._native").origin
        record["ppm3_library_path"] = native_path
        record["ppm3_library_sha256"] = digest(native_path)
        record["ppm3_residue_library_sha256"] = digest(ROOT / "crimm/Modeller/_ppm3/res.lib")
    cell = (32.0, 32.0, 80.0) if args.case == "control" else (40.0, 40.0, 80.0)
    if preset and preset["kind"] == "biological":
        cell = tuple(preset["reference_box_A"])
    if getattr(args, "cell", None) is not None:
        cell = tuple(args.cell)
    if npt_steps and not np.isclose(cell[0], cell[1], rtol=0, atol=1e-8):
        raise ValueError("The semi-isotropic NPT trial requires equal nominal X and Y dimensions.")
    record["cell_A"] = cell
    if preset:
        spec = MembraneSpec.from_preset(preset_name, box_xy=cell[:2], box_z=cell[2],
            cholesterol_percent=getattr(args, "cholesterol_percent", None),
            initial_area_scale=getattr(args, "initial_area_scale", 1.0),
            random_seed=args.seed, max_repack_attempts=200)
        record["membrane_preset"] = preset
    else:
        spec = MembraneSpec(box_xy=cell[:2], box_z=cell[2], lipid_ratios={"POPC": 1.0},
                        upper_lipid_ratios=args.upper_lipid_ratios, lower_lipid_ratios=args.lower_lipid_ratios,
                        area_per_lipid=None if args.upper_lipid_ratios or args.lower_lipid_ratios else 64.0,
                        initial_area_scale=getattr(args, "initial_area_scale", 1.0),
                        lipid_z=20.0, membrane_core_z=15.0, random_seed=args.seed, max_repack_attempts=200)
    record["membrane_core_z_A"] = spec.membrane_core_z
    record["packing_limits"] = {"conformer_retries": spec.max_repack_attempts,
                                "protein_clash_cutoff_A": spec.protein_clash_cutoff,
                                "membrane_clash_cutoff_A": spec.membrane_clash_cutoff}
    builder = MembraneBuilder(model, spec)
    record["initial_preparation"] = {"nominal_cell_A": cell,
        "area_scale": spec.initial_area_scale, "lipid_counts_fixed_before_expansion": True}
    spec = builder.spec
    cell = tuple(builder._box_dims)
    record["cell_A"] = cell
    if builder.topology is not model.topology_loader:
        raise ValueError("The membrane builder does not share the model topology loader.")
    builder.prepare_topology()
    builder.add_lipid_bilayer()
    record["leaflet_counts"] = [builder.upper_leaflet, builder.lower_leaflet]
    record["leaflet_composition"] = builder.leaflet_composition
    record["lipid_geometry"] = {name: {key: template[key] for key in
        ("head_atoms", "chiral_centres", "dihedral_guards")}
        for name, template in builder.lipid_library.templates.items()
        if name in builder.upper_leaflet or name in builder.lower_leaflet}
    record["initial_geometry"] = geometry_checks(model, builder.lipid_library, cell)
    save_record(directory, record, "solvation")
    if not args.dry:
        builder.solvate()
        save_record(directory, record, "ions")
        builder.add_ions()
        record["solvation"] = dict(model._solvation_info)
        fixed = model._solvation_info.get("fixed_ion_counts", {})
        record["actual_ion_counts"] = {name: sum(r.resname == name for r in model.get_residues())
                                      for name in (spec.cation, spec.anion, *fixed)}
        expected_ions = {spec.cation: model._solvation_info.get("n_cation", 0),
                         spec.anion: model._solvation_info.get("n_anion", 0), **fixed}
        if record["actual_ion_counts"] != expected_ions:
            raise ValueError("Actual ion counts differ from the calculated counts.")
    save_record(directory, record, "construction_validation")
    record["initial_geometry"] = geometry_checks(model, builder.lipid_library, cell, spec.membrane_core_z)
    geometry = record["initial_geometry"]
    if (geometry["reversed_lipids"] or geometry["trans_lipids"] or geometry["inward_lipid_heads"]
            or geometry["ring_penetrations"] or geometry["waters_in_core"] or geometry["ions_in_core"]
            or geometry["periodic_heavy_contacts_below_1_2_A"]):
        raise ValueError("Initial membrane geometry did not meet the acceptance criteria.")
    if geometry["minimum_contact_A"] == 0:
        raise ValueError("Initial atoms coincide exactly across a periodic cell boundary.")
    if not args.dry and (geometry["periodic_water_face_gaps_A"] is None
                        or max(geometry["periodic_water_face_gaps_A"]) > 2.0):
        raise ValueError("Generated water leaves an empty strip at the periodic cell faces.")
    if not args.dry and (geometry["minimum_water_oxygen_to_solute_A"] < spec.water_solvcut - 1e-8
                        or geometry["minimum_water_oxygen_pair_A"] < PERIODIC_WATER_OO_CUTOFF - 1e-8):
        raise ValueError("Generated water does not meet the periodic separation cutoffs.")
    record["counts"] = {"lipids": builder.count_lipids(), "sterols": builder.count_sterols(), "water": builder.count_waters(), "ions": builder.count_ions()}
    record["scope"] = "dry diagnostic, not solvated-membrane acceptance" if args.dry else "solvated membrane"
    issues = validate_psf(model, strict=True)
    if issues:
        raise ValueError(f"PSF validation failed: {issues[:5]}")
    for residue in model.get_residues():
        if residue.missing_atoms or residue.missing_hydrogens:
            raise ValueError(f"Incomplete residue {residue.full_id}.")
    write_psf(model, str(directory / "system.psf"))
    write_crd(model, str(directory / "system.crd"))
    toppar = topo.write_toppar(path=str(directory))
    psf, _, _ = coordinate_pair(directory)
    record["protein_segments"] = sorted({residue.segid.strip() for chain in model
        if chain.chain_type == "Polypeptide(L)" for residue in chain})
    record["natom"] = len(psf.atoms)
    record["charge_e"] = sum(a.charge for a in psf.atoms)
    if abs(record["charge_e"]) > 1e-4:
        raise ValueError("The exported system is not neutral.")
    record["topology_counts"] = {name: len(getattr(psf, name)) for name in ("bonds", "angles", "dihedrals", "impropers", "cmap")}
    paths = [directory / "system.psf", directory / "system.crd", *map(Path, toppar)]
    record["input_sha256"] = {str(p.relative_to(directory)): digest(p) for p in paths}
    return model, builder


def nonbonded_script(cell, short_range=False):
    """Keep the optional local warmup separate from the full PME Hamiltonian."""
    cutoffs = "ctonnb 5 ctofnb 6 cutnb 8 cutim 8" if short_range else "ctonnb 10 ctofnb 12 cutnb 16 cutim 16"
    electrostatics = ("noewald nopme" if short_range else
        f"ewald pmew fftx {checkfft(np.ceil(cell[0]), margin=0)} "
        f"ffty {checkfft(np.ceil(cell[1]), margin=0)} fftz {checkfft(np.ceil(cell[2]), margin=0)} "
        "kappa .34 spline order 6")
    return (f"nbonds atom vatom vfswitch {cutoffs} inbfrq -1 imgfrq -1 "
            f"wmin 1 cdie eps 1 {electrostatics}\n")


def relaxation_script(phase, steps, temperature, seed, piston_mass, previous=None):
    """Generate a 1-fs NVT or NPT pilot with restart and trajectory output."""
    frequency = min(100, steps)
    script = f"open write unit 12 card name {phase}.rst\nopen write unit 13 file name {phase}.dcd\n"
    if previous is not None:
        script += f"open read unit 11 card name {previous}.rst\n"
    integrator = "langevin" if phase == "nvt" else "cpt"
    options = [f"dyna leap {integrator} {'restart' if previous else 'start'}",
               f"nstep {steps} timestep 0.001",
               f"iunread {11 if previous else -1} iunwrite 12 iuncrd 13 iunvelo -1",
               f"nsavc {frequency} nsavvelo 0 nprint {frequency} iprfrq {frequency}",
               f"isvfrq {frequency} inbfrq -1 imgfrq -1 ihbfrq 0 ilbfrq 0",
               "ntrfrq 5000 ieqfrq 0 ihtfrq 0 ichecw 0",
               f"firstt {temperature} finalt {temperature} tbath {temperature}"]
    if previous is None:
        options.append(f"iasvel 1 tstruc {temperature}")
        seeds = np.random.default_rng(seed).integers(1, 2**31 - 1, 4)
        options.append("iseed " + " ".join(map(str, seeds)))
    if phase == "npt":
        options += [f"pcons pint pref 1.0 pmass {piston_mass} pgamma 20",
                    f"hoover reft {temperature} tmass 2000"]
    script += " -\n".join(options) + "\n"
    return script


def charmm_worker(directory, record):
    for name, expected in record["input_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Input hash changed: {name}.")
    for name, expected in record["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Source hash changed: {name}.")
    if "ppm3_library_path" in record and digest(record["ppm3_library_path"]) != record["ppm3_library_sha256"]:
        raise ValueError("The native PPM3 library hash changed.")
    data, entries, input_coords = coordinate_pair(directory)
    import pycharmm as pcm
    from pycharmm import coor, energy, psf, read, settings, write
    pcm.lingo.set_default_error_handling(True)
    settings.set_bomb_level(0)
    save_record(directory, record, "charmm_load")
    read.stream("load_toppar.str")
    settings.set_bomb_level(0)
    read.psf_card("system.psf")
    read.coor_card("system.crd")
    if psf.get_natom() != len(entries):
        raise ValueError("CHARMM atom count differs from the exported PSF.")
    np.testing.assert_allclose(psf.get_charges(), [a.charge for a in data.atoms], atol=1e-8, rtol=0)
    np.testing.assert_allclose(psf.get_amass(), [a.mass for a in data.atoms], atol=1e-6, rtol=0)
    selection = pcm.SelectAtoms().all_atoms()
    native_ids = list(zip(selection.get_seg_ids(), selection.get_res_ids(), selection.get_res_names(), selection.get_atom_types()))
    if native_ids != [(a.segid, a.resid, a.resname, a.atomname) for a in data.atoms]:
        raise ValueError("CHARMM atom identities differ from the exported PSF.")
    np.testing.assert_allclose(coor.get_positions().to_numpy(), input_coords, atol=1e-9, rtol=0)
    write.psf_card("loaded.psf", xplor=True)
    loaded = read_psf(str(directory / "loaded.psf"))
    if [a.atomtype for a in loaded.atoms] != [a.atomtype for a in data.atoms]:
        raise ValueError("CHARMM changed the force-field atom types.")
    for name in record["topology_counts"]:
        if sorted(getattr(loaded, name)) != sorted(getattr(data, name)):
            raise ValueError(f"CHARMM changed the exported {name} inventory.")
    a, b, c = record["cell_A"]
    crystal = "tetragonal" if record.get("relaxation", {}).get("npt_steps") else "orth"
    full_nonbonded = nonbonded_script((a, b, c))
    script = (f"crystal define {crystal} {a} {b} {c} 90 90 90\n"
              "crystal build cutoff 16 noper 0\n"
              "image byresid xcen 0 ycen 0 zcen 0 sele all end\n" + full_nonbonded)
    (directory / "periodic.str").write_text(script)
    pcm.lingo.charmm_script(script)
    def snapshot():
        values = {str(k): float(v) for k, v in energy.get_energy().iloc[0].to_dict().items()}
        gradient = coor.get_forces().to_numpy()
        if not np.isfinite(list(values.values())).all() or not np.isfinite(gradient).all():
            raise ValueError("CHARMM returned nonfinite energy or gradients.")
        return values, gradient
    save_record(directory, record, "energy")
    record["initial_unrestrained"], _ = snapshot()
    initial = coor.get_positions().copy()
    write.coor_card("restraint_reference.crd")
    record["restraint_reference_crd_sha256"] = digest(directory / "restraint_reference.crd")
    save_record(directory, record, "restraints")
    pcm.lingo.charmm_script("coor copy comp")
    metadata = record.get("lipid_geometry", {})
    if metadata:
        head_selection = pcm.SelectAtoms(selection=np.array([
            atom.resname in metadata and atom.atomname in metadata[atom.resname]["head_atoms"]
            for atom in data.atoms], dtype=bool))
    else:
        head_selection = pcm.SelectAtoms().by_atom_type("P") | (pcm.SelectAtoms().by_seg_id("CHOL") & pcm.SelectAtoms().by_atom_type("O3"))
    head_selection.store("MHEAD")
    restrained = head_selection
    if record["case"] == "helix":
        for segment in record["protein_segments"]:
            restrained |= pcm.SelectAtoms().by_seg_id(segment) & ~pcm.SelectAtoms().all_hydrogen_atoms()
    record["restrained_atoms"] = restrained.get_atom_indexes()
    if not record["restrained_atoms"]:
        raise ValueError("The harmonic restraint selection is empty.")
    record["reference_sha256"] = hashlib.sha256(initial.to_numpy().tobytes()).hexdigest()
    record["lipid_restraint_axes"] = "Z"
    record["protein_restraint_axes"] = "XYZ"
    record["target_grms"] = 1.0
    lipids = sorted({(atom.segid, atom.resid, atom.resname) for atom in data.atoms
                     if atom.resname in (metadata or MembraneBuilder.LIPID_TAILS)})
    dihedrals = "cons cldh\n"
    restraint_count = 0
    expected_guard_energy = 0.0
    coordinates_by_identity = {(atom.segid, atom.resid, atom.atomname): coord
                               for atom, coord in zip(data.atoms, initial.to_numpy())}
    for segment, residue, resname in lipids:
        guards = [(entry["names"], entry["angle"], entry["width"])
                  for entry in metadata.get(resname, {}).get("dihedral_guards", [])]
        if not metadata:
            guards = [(names, 0.0, 0.0) for names in CIS_BONDS.get(resname, ())]
        if not metadata and resname != "CHL1":
            guards.append((("C3", "C1", "C2", "O21"), 120.0, 2.5))
        elif not metadata:
            for centre, (first, second, last) in CHIRAL_CENTRES[resname].items():
                guards.append(((first, second, centre, last), 120.0 * STEROL_CHIRAL_SIGNS[centre], 10.0))
        for names, angle, width in guards:
            atoms = " ".join(f"{segment} {residue} {name}" for name in names)
            dihedrals += f"cons dihe force 250 min {angle} width {width} {atoms}\n"
            restraint_count += 1
            actual_angle = dihedral_angle([coordinates_by_identity[(segment, residue, name)] for name in names])
            difference = abs((actual_angle - angle + 180) % 360 - 180)
            expected_guard_energy += 250 * np.radians(max(difference - width, 0)) ** 2
    (directory / "lipid_dihedral_guard.str").write_text(dihedrals)
    pcm.lingo.charmm_script(dihedrals)
    actual_guard_energy = snapshot()[0]["CDIH"]
    np.testing.assert_allclose(actual_guard_energy, expected_guard_energy, atol=1e-5, rtol=1e-8)
    record["dihedral_guard_probe_energy"] = actual_guard_energy
    record["lipid_dihedral_restraints"] = {"count": restraint_count, "force": 250.0,
        "cis_min_deg": 0.0, "chirality_min_deg": 120.0, "chirality_width_deg": 2.5,
        "sterol_chirality_min_deg": [-120.0, 120.0], "sterol_chirality_width_deg": 10.0}
    if metadata:
        record["lipid_dihedral_restraints"] = {"count": restraint_count, "force": 250.0,
                                               "definitions": "lipid_geometry"}
    if not record["dry"]:
        # Initialization-only soft guard, with 2 A for O-H extent and soft penetration.
        radius = record["membrane_core_z_A"] + 2.0
        water_force = 10.0
        record["water_core_restraint"] = {"force": water_force, "oxygen_plane_distance_A": radius}
        guard = ("mmfp\ngeo plane xref 0 yref 0 zref 0 zdir 1 "
                 f"harmonic outside force {water_force} droff {radius} "
                 "select resname TIP3 .and. type OH2 end\nend\n")
        (directory / "water_guard.str").write_text(guard)
        pcm.lingo.charmm_script(guard)
    save_record(directory, record, "minimization")
    def set_restraint(force):
        commands = (f"cons harm clear\ncons harm absolute comp force {force} "
                    "xscale 0 yscale 0 zscale 1 sele MHEAD end\n")
        if record["protein_segments"]:
            protein = " .or. ".join(f"segid {segment}" for segment in record["protein_segments"])
            commands += (f"cons harm absolute comp force {force} xscale 1 yscale 1 zscale 1 "
                         f"sele ({protein}) .and. .not. hydrogen end\n")
        pcm.lingo.charmm_script(commands)
    record["stages"] = []
    try:
        set_restraint(0.1)
        displaced = initial.copy()
        displaced.iloc[record["restrained_atoms"][0], 2] += 0.1
        coor.set_positions(displaced)
        probe, _ = snapshot()
        np.testing.assert_allclose(probe["HARM"], 0.001, atol=1e-8, rtol=0)
        record["restraint_probe_energy"] = probe["HARM"]
        if not record["dry"]:
            oxygen_indexes = [i for i, atom in enumerate(data.atoms) if atom.resname == "TIP3" and atom.atomname == "OH2"]
            depth = np.maximum(radius - np.abs(initial.iloc[oxygen_indexes, 2].to_numpy()), 0)
            np.testing.assert_allclose(probe["GEO"], 0.5 * water_force * np.dot(depth, depth), atol=1e-8, rtol=0)
            record["water_core_probe_energy"] = probe["GEO"]
        coor.set_positions(initial)
        record["initial_final_hamiltonian"], _ = snapshot()
        save_record(directory, record, "hydrogen_relaxation")
        heavy_indexes = (~pcm.SelectAtoms().all_hydrogen_atoms()).get_atom_indexes()
        before_hydrogens = coor.get_positions().to_numpy()
        try:
            if record.get("short_range_warmup"):
                pcm.lingo.charmm_script(nonbonded_script((a, b, c), short_range=True))
            pcm.lingo.charmm_script("cons fix sele .not. hydrogen end\nmini sd nstep 500 tolgr 1.0 tole 0.0")
        finally:
            pcm.lingo.charmm_script("cons fix sele none end")
            if record.get("short_range_warmup"):
                pcm.lingo.charmm_script(full_nonbonded)
            pcm.lingo.charmm_script("update")
        displacement = coor.get_positions().to_numpy()[heavy_indexes] - before_hydrogens[heavy_indexes]
        displacement -= np.array(record["cell_A"]) * np.rint(displacement / record["cell_A"])
        np.testing.assert_allclose(displacement, 0, atol=1e-8, rtol=0)
        record["hydrogen_relaxation"] = {"sd_limit": 500, "fixed_heavy_atoms": len(heavy_indexes),
            "short_range_warmup": bool(record.get("short_range_warmup")), "full_PME_restored": True,
            "maximum_heavy_displacement_A": float(np.abs(displacement).max()), "energy_after_release": snapshot()[0]}
        save_record(directory, record, "minimization")
        for force, sd_steps, abnr_steps in ((10.0, 500, 0), (1.0, 500, 2000), (0.1, 500, 2000)):
            set_restraint(force)
            pcm.lingo.charmm_script(f"mini sd nstep {sd_steps} tolgr 1.0 tole 0.0")
            if abnr_steps:
                pcm.lingo.charmm_script(f"mini abnr nstep {abnr_steps} tolgr 1.0 tole 0.0")
            values, _ = snapshot()
            record["stages"].append({"force": force, "sd_limit": sd_steps, "abnr_limit": abnr_steps, "energy": values})
            save_record(directory, record, "minimization")
        record["final_final_hamiltonian"] = values
    finally:
        pcm.lingo.charmm_script("cons fix sele none end\ncons harm clear\ncons cldh\nmmfp\ngeo reset\nend")
    final = coor.get_positions().copy()
    if not np.isfinite(final.to_numpy()).all():
        raise ValueError("CHARMM produced nonfinite final coordinates.")
    write.coor_card("minimized.crd")
    _, _, reloaded = coordinate_pair(directory, "minimized.crd")
    np.testing.assert_allclose(reloaded, final.to_numpy(), atol=5e-5, rtol=0)
    reference_energy, reference_gradient = snapshot()
    np.testing.assert_allclose([reference_energy[name] for name in ("HARM", "CDIH", "GEO")], 0, atol=1e-12, rtol=0)
    record["final_unrestrained"] = reference_energy
    record["periodic_translations"] = []
    for resname in dict.fromkeys((*MembraneBuilder.LIPID_TAILS, *metadata, "TIP3", "POT", "CLA", "CAL")):
        atom = next((atom for atom in data.atoms if atom.resname == resname), None)
        if atom is None:
            continue
        indexes = [i for i, entry in enumerate(data.atoms) if (entry.segid, entry.resid) == (atom.segid, atom.resid)]
        shifted = final.copy()
        shifted.iloc[indexes, 0] += a
        coor.set_positions(shifted)
        pcm.lingo.charmm_script("update")
        values, gradient = snapshot()
        np.testing.assert_allclose(values["ENER"], reference_energy["ENER"], atol=1e-4, rtol=1e-9)
        np.testing.assert_allclose(gradient, reference_gradient, atol=1e-4, rtol=1e-8)
        record["periodic_translations"].append(resname)
        coor.set_positions(final)
        pcm.lingo.charmm_script("update")
    record["minimization_converged"] = record["final_final_hamiltonian"]["GRMS"] <= 1.0
    record["same_hamiltonian_energy_decreased"] = record["final_final_hamiltonian"]["ENER"] <= record["initial_final_hamiltonian"]["ENER"]
    record["charmm_diagnostic_passed"] = record["minimization_converged"] and record["same_hamiltonian_energy_decreased"]
    record["minimized_crd_sha256"] = digest(directory / "minimized.crd")
    record["validated_crd"] = "minimized.crd"
    if "relaxation" in record and record["charmm_diagnostic_passed"]:
        from MDAnalysis.coordinates.DCD import DCDReader
        trial = record["relaxation"]
        trial.update(pressure_atm=1.0 if trial["npt_steps"] else None,
                     pressure_coupling="tetragonal XY coupled, Z independent" if trial["npt_steps"] else None,
                     restraints="NVT: head Z, water core, stereo. NPT: stereo only.", stages=[])
        pcm.lingo.charmm_script("coor copy comp\nshake bonh param fast")
        phosphate = np.array([i for i, atom in enumerate(data.atoms)
                              if abs(atom.mass - 30.9738) < 0.02 and
                              atom.resname in (metadata or MembraneBuilder.LIPID_TAILS)], dtype=int)
        upper_phosphate = phosphate[input_coords[phosphate, 2] >= 0]
        lower_phosphate = phosphate[input_coords[phosphate, 2] < 0]
        water_atoms = np.array([i for i, atom in enumerate(data.atoms) if atom.resname == "TIP3"], dtype=int)
        previous = None
        water_guard_active = False
        try:
            for phase in ("nvt", "npt"):
                steps = trial[f"{phase}_steps"]
                if not steps:
                    continue
                save_record(directory, record, phase)
                if phase == "nvt":
                    set_restraint(0.1)
                    pcm.lingo.charmm_script(guard)
                    water_guard_active = True
                    pcm.lingo.charmm_script("scalar fbeta set 3.0 sele .not. hydrogen end")
                else:
                    pcm.lingo.charmm_script("cons harm clear\nscalar fbeta set 0.0")
                    if water_guard_active:
                        pcm.lingo.charmm_script("mmfp\ngeo reset\nend")
                        water_guard_active = False
                pcm.lingo.charmm_script(dihedrals.replace("force 250", f"force {250 if phase == 'nvt' else 50}"))
                command = relaxation_script(phase, steps, record["temperature_K"], record["seed"],
                                            max(int(sum(atom.mass for atom in data.atoms) / 50), 1), previous)
                (directory / f"{phase}.inp").write_text(command)
                # File continuation avoids truncation in the native inline-command buffer.
                pcm.lingo.charmm_script(f"stream {phase}.inp")
                values, _ = snapshot()
                cell = [float(pcm.lingo.get_energy_value(name)) for name in ("XTLA", "XTLB", "XTLC")]
                if not np.isfinite(cell).all() or min(cell) <= 0:
                    raise ValueError(f"The {phase} trial produced invalid box dimensions.")
                samples = []
                phosphate_separation, core_water_atoms = [], []
                with DCDReader(str(directory / f"{phase}.dcd")) as trajectory:
                    if trajectory.n_atoms != len(data.atoms):
                        raise ValueError("The dynamics trajectory atom count changed.")
                    for frame in trajectory:
                        if not np.isfinite(frame.positions).all() or not np.isfinite(frame.dimensions).all():
                            raise ValueError("The dynamics trajectory contains nonfinite values.")
                        samples.append([float(frame.time), *map(float, frame.dimensions[:3])])
                        z = frame.positions[:, 2].astype(float)
                        z -= frame.dimensions[2] * np.rint(z / frame.dimensions[2])
                        core_water_atoms.append(int(np.count_nonzero(np.abs(z[water_atoms]) < record["membrane_core_z_A"])))
                        if len(upper_phosphate) and len(lower_phosphate):
                            phosphate_separation.append(float(z[upper_phosphate].mean() - z[lower_phosphate].mean()))
                if phase == "npt" and not np.allclose(np.array(samples)[:, 1], np.array(samples)[:, 2], atol=1e-5):
                    raise ValueError("The NPT trial lost tetragonal XY symmetry.")
                write.coor_card(f"{phase}.crd")
                trial["stages"].append({"ensemble": phase.upper(), "steps": steps,
                    "time_ps": steps * trial["timestep_ps"], "final_cell_A": cell,
                    "energy": values, "box_samples_time_ps_A": samples,
                    "phosphate_plane_separation_A": phosphate_separation,
                    "core_water_atom_counts": core_water_atoms,
                    "input_sha256": digest(directory / f"{phase}.inp"),
                    "trajectory_sha256": digest(directory / f"{phase}.dcd"),
                    "restart_sha256": digest(directory / f"{phase}.rst")})
                record["final_cell_A"] = cell
                record["validated_crd"] = f"{phase}.crd"
                previous = phase
                save_record(directory, record, phase)
        finally:
            pcm.lingo.charmm_script("cons harm clear\ncons cldh")
            if water_guard_active:
                pcm.lingo.charmm_script("mmfp\ngeo reset\nend")
        record["relaxation_unrestrained_energy"], _ = snapshot()
        record["relaxation_completed"] = True
    save_record(directory, record, "charmm_complete")


def lipid_ratios(value):
    def unique(pairs):
        result = {}
        for name, ratio in pairs:
            if name in result:
                raise ValueError(f"Duplicate lipid name: {name}.")
            result[name] = ratio
        return result
    try:
        result = json.loads(value, object_pairs_hook=unique)
        if not isinstance(result, dict):
            raise ValueError("Lipid ratios must be a JSON object.")
        return result
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("control", "helix"), default="control")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--upper-lipid-ratios", type=lipid_ratios)
    parser.add_argument("--lower-lipid-ratios", type=lipid_ratios)
    parser.add_argument("--membrane", choices=list_membrane_presets())
    parser.add_argument("--cholesterol-percent", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--cell", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--initial-area-scale", type=float, default=1.0,
                        help="Expand XY area after resolving nominal lipid counts. 1.05 adds 5%% area.")
    parser.add_argument("--nvt-steps", type=int, default=0, help="Optional 1-fs NVT pilot after minimization.")
    parser.add_argument("--npt-steps", type=int, default=0, help="Optional 1-fs semi-isotropic NPT pilot. Requires square XY.")
    parser.add_argument("--short-range-warmup", action="store_true",
                        help="Use a 6-A non-PME hydrogen warmup, then restore full PME for minimization and dynamics.")
    parser.add_argument("--ppm-membrane", choices=PPM3_MEMBRANE_TYPES)
    parser.add_argument("--topology", choices=("in", "out"), default="in")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--charmm-lib", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="crimm-charmm-membrane-"))
    if args.output and not args.worker:
        directory.mkdir(parents=True, exist_ok=False)
    record = {"case": args.case, "seed": args.seed, "dry": args.dry, "status": "running"}
    try:
        if args.worker:
            record = json.loads((directory / "result.json").read_text())
            charmm_worker(directory, record)
            return
        print(directory, flush=True)
        record["charmm_library"] = str(args.charmm_lib.resolve() / "libchmm.so")
        record["charmm_library_sha256"] = digest(record["charmm_library"])
        record["python"] = sys.version
        record["source_sha256"] = {name: digest(ROOT / name) for name in (
            "scripts/membrane_pycharmm.py", "crimm/Modeller/MembraneBuilder.py",
            "crimm/Modeller/Solvator.py", "crimm/Modeller/TopoLoader.py",
            "crimm/Modeller/MembraneOrienter.py", "crimm/charmm_gui_9qsy/pentest.py",
            "crimm/Data/lipid_conformers.npz", "crimm/IO/PSFWriter.py", "crimm/IO/CRDWriter.py")}
        if args.membrane:
            for name in ("crimm/Modeller/MembranePresets.py", "crimm/Data/oprlm_lipids.npz",
                         "crimm/Data/oprlm_lipids.json", "crimm/charmm_gui_9qsy/checkfft.py"):
                record["source_sha256"][name] = digest(ROOT / name)
        for native_source in (ROOT / "crimm/Modeller/_ppm3").glob("*.f*"):
            record["source_sha256"][str(native_source.relative_to(ROOT))] = digest(native_source)
        record["source_sha256"]["crimm/Modeller/_ppm3/res.lib"] = digest(ROOT / "crimm/Modeller/_ppm3/res.lib")
        save_record(directory, record, "construction")
        model, builder = prepare(directory, args, record)
        save_record(directory, record, "charmm_start")
        env = dict(os.environ, CHARMM_LIB_DIR=str(args.charmm_lib.resolve()),
                   NUMBA_CACHE_DIR=str(directory / "numba-cache"), OMP_NUM_THREADS="1")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(directory),
                   "--charmm-lib", str(args.charmm_lib), "--case", args.case]
        timeout = 3600 if args.membrane or args.nvt_steps or args.npt_steps else 900
        with (directory / "charmm.log").open("w") as log:
            try:
                result = subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                record = json.loads((directory / "result.json").read_text())
                record["charmm_timeout_seconds"] = timeout
                raise
        record = json.loads((directory / "result.json").read_text())
        record["charmm_exit_code"] = result.returncode
        if result.returncode or record["stage"] != "charmm_complete":
            raise RuntimeError(f"The CHARMM child failed at {record['stage']}. Read charmm.log.")
        save_record(directory, record, "final_validation")
        final_crd = record.get("validated_crd", "minimized.crd")
        _, entries, coords = coordinate_pair(directory, final_crd)
        by_identity = {(r.segid.strip(), str(r.id[1]), r.resname, atom.name): atom for r in model.get_residues() for atom in r}
        for entry, coord in zip(entries, coords):
            by_identity[(entry.segid, str(entry.resid), entry.resname, entry.atomname)].coord = coord
        record["final_geometry"] = geometry_checks(model, builder.lipid_library,
            record.get("final_cell_A", record["cell_A"]), record["membrane_core_z_A"])
        initial_geometry = record["initial_geometry"]
        final_geometry = record["final_geometry"]
        record["construction_checks_passed"] = (not args.dry and record["counts"]["water"] > 0
            and record["counts"]["lipids"] + record["counts"]["sterols"] == sum(sum(leaf.values()) for leaf in record["leaflet_counts"])
            and not initial_geometry["reversed_lipids"] and not final_geometry["reversed_lipids"]
            and not final_geometry["trans_lipids"] and not final_geometry["inward_lipid_heads"]
            and not final_geometry["waters_in_core"] and not final_geometry["ions_in_core"]
            and final_geometry["periodic_water_face_gaps_A"] is not None
            and max(final_geometry["periodic_water_face_gaps_A"]) <= 2.0
            and final_geometry["actual_leaflet_composition"] == [{name: count for name, count in leaf.items() if count} for leaf in record["leaflet_counts"]]
            and not initial_geometry["ring_penetrations"] and not final_geometry["ring_penetrations"]
            and final_geometry["periodic_contacts_below_1_2_A"] == 0)
        record["case_acceptance_passed"] = record["construction_checks_passed"] and record["charmm_diagnostic_passed"]
        if "relaxation" in record:
            record["case_acceptance_passed"] &= record.get("relaxation_completed", False)
        if args.case == "helix":
            record["protein_geometry"] = protein_checks(directory, record["protein_segments"], record["orientation"], final_crd)
            record["case_acceptance_passed"] &= record["protein_geometry"]["passed"]
        record["full_membrane_acceptance_matrix"] = "not run"
        if not record["charmm_diagnostic_passed"]:
            raise RuntimeError("The CHARMM minimization did not meet the energy and gradient criteria.")
        if not args.dry and not record["construction_checks_passed"]:
            raise RuntimeError("The membrane construction did not meet the geometry criteria.")
        if args.case == "helix" and not record["protein_geometry"]["passed"]:
            raise RuntimeError("The protein did not retain its structure and membrane placement.")
        record["status"] = "dry_diagnostic" if args.dry else "passed"
        save_record(directory, record, "complete")
        print(json.dumps({key: record[key] for key in ("status", "case_acceptance_passed", "charmm_diagnostic_passed", "construction_checks_passed")}), flush=True)
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        save_record(directory, record, record.get("stage", "startup"))
        raise


if __name__ == "__main__":
    main()
