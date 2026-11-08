"""Compile whole OPRLM lipid molecules without solvent or repeated text structures.

Run import_oprlm_reference.py --download-archives --audit first. Compound lipids
retain every source atom, charge, mass, and covalent connection. The identity map
records the original residues when a compound needs unique CHARMM atom names.
"""

import argparse
from collections import Counter, defaultdict
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crimm.IO.CRDParser import CRDParser
from crimm.IO.PSFReader import read_psf


COMPOUNDS = {
    ("BGAL", "CER160"): "GALCER",
    ("BGLC", "CER160"): "GLCCER",
    ("BMAN", "CER160"): "MIPC",
    ("BGAL", "BGAL", "DIPA"): "DGDGLI",
    ("BGAL", "DIPA"): "MGDGLI",
    ("BGAL", "BGAL", "PLEPA"): "DGDGPL",
    ("BGAL", "PLEPA"): "MGDGPL",
    ("BQUI", "PLEPA"): "SQDGPL",
    ("BGAL", "BGAL", "POPA"): "DGDGPO",
    ("BGAL", "POPA"): "MGDGPO",
    ("BQUI", "POPA"): "SQDGPO",
    ("BQUI", "DPPA"): "SQDGDP",
    ("BGAL", "BGAL", "DLPA"): "DGDGDL",
    ("BGAL", "DLPA"): "MGDGDL",
    ("BQUI", "DLPA"): "SQDGDL",
    tuple(sorted(["ECLIPA", *["AKDO"] * 2, *["AHEP"] * 4,
                  *["AGLC"] * 3, "AGAL"])): "ECLIPA",
}
ALIASES = {
    "PMm": {"CMH": "GALCER"},
    "PMp": {"CMH": "GLCCER", "DGDG": "DGDGLI", "LLPI": "LINPI"},
    "VAC": {"CMH": "GLCCER", "DGDG": "DGDGDL", "MGDG": "MGDGDL", "SQDG": "SQDGDL"},
    "THYp": {"DGDG-PLE": "DGDGPL", "DGDG-DLI": "DGDGLI", "MGDG-PLE": "MGDGPL",
             "MGDG-DLI": "MGDGLI", "SQDG-PLE": "SQDGPL"},
    "THYb": {"DGDG-PO": "DGDGPO", "MGDG-PO": "MGDGPO", "SQDG-PO": "SQDGPO", "SQDG-DP": "SQDGDP"},
    "ENDm": {"PDPE": "PDOPE"},
    "LYSm": {"BMP": "BMGP"},
}
STEROIDS = {"CHL1", "ERG", "SITO", "STIG", "CAMP", "GSITO", "GCAMP", "GSTIG",
            "PGSITO", "PGCAMP", "PGSTIG"}
ELEMENT_MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.9994, "P": 30.974, "S": 32.06}


def reference_forcefield(archive_path, atomtypes):
    masses, doubles, parameters, internal_coordinates = {}, defaultdict(set), [], defaultdict(list)
    with tarfile.open(archive_path) as archive:
        members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
        loader = archive.extractfile(members["toppar.str"]).read().decode("utf-8")
        order = dict.fromkeys(re.findall(r"toppar/([\w.-]+)", loader))
        for name in order:
            base = name in {"top_all36_lipid.rtf", "top_all36_carb.rtf", "top_all36_cgenff.rtf",
                            "par_all36_lipid.prm", "par_all36_carb.prm", "par_all36_cgenff.prm",
                            "top_all36_prot.rtf", "par_all36m_prot.prm", "top_all36_na.rtf", "par_all36_na.prm"}
            extension = name.startswith("toppar_all36_lipid_") or name in {
                "toppar_all36_carb_glycolipid.str", "toppar_all36_carb_imlab.str"}
            if not (base or extension):
                continue
            member = members[name]
            text = archive.extractfile(member).read().decode("utf-8")
            residue = None
            for line in text.splitlines():
                fields = line.split("!", 1)[0].split()
                if not fields:
                    continue
                if fields[0].upper() == "MASS":
                    masses[fields[2]] = float(fields[3])
                elif fields[0].upper() in {"RESI", "PRES"}:
                    residue = fields[1]
                elif fields[0].upper() in {"DOUB", "DOUBLE"} and residue is not None:
                    doubles[residue].update(frozenset(pair) for pair in zip(fields[1::2], fields[2::2]))
                elif fields[0].upper() == "IC" and len(fields) == 10 and residue is not None:
                    internal_coordinates[residue].append((fields[1:5], [float(value) for value in fields[5:]]))
            blocks = [text] if name.endswith(".prm") else re.findall(
                r"^\s*read para[^\n]*\n(.*?)^\s*end\s*$", text, re.I | re.M | re.S)
            if name in {"par_all36m_prot.prm", "par_all36_na.prm"}:
                from crimm.IO.PRMParser import categorize_lines

                categories = categorize_lines(text.splitlines())
                shared = []
                for section, width in (("bonds", 2), ("angles", 3), ("dihedrals", 4),
                                       ("improper", 4), ("nonbonded", 1), ("nbfix", 2)):
                    shared.append(section.upper())
                    for line in categories[section]:
                        fields = line.split("!", 1)[0].split()
                        required = set(fields[:width])
                        if (required & atomtypes if section == "nbfix" else required <= atomtypes | {"X"}):
                            shared.append(line)
                blocks = ["\n".join(shared)]
            for block in blocks:
                parameters.extend(line.rstrip() for line in block.splitlines()
                                  if line.strip() and line.strip().upper() != "END"
                                  and not line.lstrip().startswith("*"))
    if not masses or not parameters:
        raise ValueError("The reference archive contains no membrane force field.")
    return masses, doubles, parameters, internal_coordinates


def placement_metadata(name, template, coordinates, restraint_text):
    source = template["source_identity"]
    atoms = template["atoms"]
    names = [atom[0] for atom in atoms]
    heads = [i for i, (_, _, original) in enumerate(source) if original == "P"]
    if name == "ECLIPA":
        heads = [i for i, (part, _, original) in enumerate(source)
                 if part == 0 and (original.startswith("CA") or original.startswith("CB"))]
    elif name in STEROIDS:
        heads = [i for i, (part, _, original) in enumerate(source)
                 if part == 0 and original in {"O3", "2O1"}]
        if not heads:
            heads = [i for i, (part, _, original) in enumerate(source) if part == 0 and original == "C3"]
    elif not heads:
        heads = [i for i, atom in enumerate(atoms) if atom[4] == "P"]
        if not heads and len({part for part, _, _ in source}) > 1:
            heads = [i for i, (part, _, original) in enumerate(source) if part == 1 and original == "C1"]
        if not heads and name in {"MEN8", "AROL"}:
            heads = [i for i, (_, _, original) in enumerate(source)
                     if original in ({"OR5", "OR8"} if name == "MEN8" else {"C2"})]
    if not heads:
        raise ValueError(f"No explicit head anchor for {name}.")
    neighbours = defaultdict(set)
    for a, b in template["bonds"]:
        neighbours[a].add(b)
        neighbours[b].add(a)
    selector_name = "LPSA" if name == "ECLIPA" else source[0][1]
    block = re.search(rf"if lipid \.eq\. {re.escape(selector_name)} then(.*?)endif", restraint_text, re.I | re.S)
    tails = []
    if block:
        if name in {"PGMAR", "PGAR", "AROL", "MEN8", "ARCL2"}:
            selection = re.search(r"define HEAD (.*?) end", block[1], re.I | re.S)
            patterns = re.findall(r"\btype\s+(\S+)", selection[1], re.I)
            heads = [i for i, (part, _, original) in enumerate(source) if part == 0 and any(
                fnmatchcase(original, pattern.replace("+", "[0-9]")) for pattern in patterns)]
        selection = re.search(r"define TAIL (.*?) end", block[1], re.I | re.S)
        if selection:
            patterns = re.findall(r"\btype\s+(\S+)", selection[1], re.I)
            tails = [i for i, (part, _, original) in enumerate(source)
                     if part == 0 and any(fnmatchcase(original, pattern) for pattern in patterns)]
    if not tails:
        distances = {index: 0 for index in heads}
        frontier = list(heads)
        for current in frontier:
            for neighbour in neighbours[current]:
                if neighbour not in distances:
                    distances[neighbour] = distances[current] + 1
                    frontier.append(neighbour)
        tails = [i for i, atom in enumerate(atoms) if atom[4] == "C" and distances.get(i, 0) > 6
                 and sum(atoms[j][4] != "H" for j in neighbours[i]) == 1]
    if not tails or coordinates[heads, 2].mean() <= coordinates[tails, 2].mean():
        raise ValueError(f"Invalid outward head/tail anchors for {name}.")
    template["head_atoms"] = [names[i] for i in heads]
    template["tail_atoms"] = [names[i] for i in tails]
    template["chain_type"] = "Sterol" if name in STEROIDS else "Lipid"
    colours = {i: atom[4] for i, atom in enumerate(atoms)}
    for _ in range(16):
        colours = {i: hashlib.sha256((colours[i] + "|" + "|".join(
            sorted(colours[j] for j in neighbours[i]))).encode()).hexdigest() for i in colours}
    centres = {}
    for centre, adjacent in neighbours.items():
        heavy = sorted(index for index in adjacent if atoms[index][4] != "H")
        if (atoms[centre][4] == "C" and len(adjacent) == 4 and len(heavy) >= 3
                and len({colours[index] for index in adjacent}) == 4):
            centres[names[centre]] = [names[index] for index in heavy[:3]]
    template["chiral_centres"] = centres


def topology_text(templates, masses, chain_type):
    lines = ["* Whole OPRLM molecules compiled from the cited reference PSFs", "*", "36 1"]
    elements = {atom[1]: atom[4] for template in templates.values() for atom in template["atoms"]}
    lines.extend(f"MASS -1 {name} {mass:.8f} {elements.get(name, '')}" for name, mass in masses.items())
    lines.append("AUTO ANGLES DIHEDRALS")
    for name, template in templates.items():
        if template["chain_type"] != chain_type:
            continue
        atoms = template["atoms"]
        lines.append(f"RESI {name} {sum(atom[2] for atom in atoms):.8f}")
        for index, atom in enumerate(atoms):
            if index == 0 or index in template["groups"]:
                lines.append("GROUP")
            lines.append(f"ATOM {atom[0]} {atom[1]} {atom[2]:.10f}")
        for key, keyword in (("bonds", "BOND"), ("impropers", "IMPR"), ("cmap", "CMAP")):
            for term in template[key]:
                record_type = "DOUBLE" if key == "bonds" and term in template["double_bonds"] else keyword
                lines.append(record_type + " " + " ".join(atoms[index][0] for index in term))
    lines.append("END")
    return "\n".join(lines) + "\n"


def stereo_guards(template, coordinates, internal_coordinates):
    from scripts.membrane_pycharmm import dihedral_angle

    coords = coordinates[0]
    names = [atom[0] for atom in template["atoms"]]
    indices = {name: index for index, name in enumerate(names)}
    source = template["source_identity"]
    source_names = {(part, original): names[index] for index, (part, _, original) in enumerate(source)}
    source_ics = []
    for part, residue in dict.fromkeys((part, residue) for part, residue, _ in source):
        for atoms, values in internal_coordinates[residue]:
            if all((part, atom.lstrip("*")) in source_names for atom in atoms):
                source_ics.append(([source_names[part, atom.lstrip("*")] for atom in atoms],
                                   values[2], atoms[2].startswith("*")))
    guards = []
    for centre, (first, second, last) in template["chiral_centres"].items():
        atoms = [first, second, centre, last]
        angle = dihedral_angle(coords[[indices[name] for name in atoms]])
        definition = "reference_coordinates"
        for ic_atoms, ic_angle, improper in source_ics:
            if improper and ic_atoms[2] == centre and set(ic_atoms) == set(atoms):
                atoms = ic_atoms
                observed = [dihedral_angle(x[[indices[name] for name in atoms]]) for x in coordinates]
                angle = observed[0]
                if any(value * angle <= 0 for value in observed):
                    angle, definition = ic_angle, "source_IC"
                template["chiral_centres"][centre] = [atoms[0], atoms[1], atoms[3]]
                break
        guards.append({"names": atoms, "angle": 120.0 if angle > 0 else -120.0,
                       "width": 10.0, "kind": "chirality", "definition": definition})
    graph = defaultdict(set)
    for a, b in template["bonds"]:
        graph[a].add(b)
        graph[b].add(a)
    for b, c in template["double_bonds"]:
        if any(template["atoms"][index][4] != "C" for index in (b, c)):
            continue
        reached, frontier = {b}, [b]
        for current in frontier:
            for neighbour in graph[current]:
                if {current, neighbour} == {b, c} or neighbour in reached:
                    continue
                reached.add(neighbour)
                frontier.append(neighbour)
        if c in reached:
            continue  # Ring double bonds do not define independent chain cis/trans isomers.
        first = sorted(i for i in graph[b] - {c} if template["atoms"][i][4] != "H")
        last = sorted(i for i in graph[c] - {b} if template["atoms"][i][4] != "H")
        if not first or not last:
            continue
        quad = [first[0], b, c, last[0]]
        angle = dihedral_angle(coords[quad])
        atoms = [names[i] for i in quad]
        definition = "reference_coordinates"
        for ic_atoms, ic_angle, improper in source_ics:
            if not improper and (ic_atoms == atoms or ic_atoms == atoms[::-1]):
                angle, definition = ic_angle, "source_IC"
                break
        guards.append({"names": atoms, "angle": 0.0 if abs(angle) < 90 else 180.0,
                       "width": 0.0, "kind": "double_bond", "definition": definition})
    template["dihedral_guards"] = guards


def valid_conformers(template, coordinates):
    """Retain raw source coordinates but select only the specified stereo states."""
    from scripts.membrane_pycharmm import dihedral_angle

    indices = {atom[0]: index for index, atom in enumerate(template["atoms"])}
    valid = []
    for index, coords in enumerate(coordinates):
        for guard in template["dihedral_guards"]:
            actual = dihedral_angle(coords[[indices[name] for name in guard["names"]]])
            if guard["kind"] == "chirality":
                passed = actual * guard["angle"] > 0
            else:
                passed = abs((actual - guard["angle"] + 180) % 360 - 180) < 90
            if not passed:
                break
        else:
            valid.append(index)
    return valid


def build_ic_conformer(name, template, masses, internal_coordinates, parameter_block):
    """Build an explicitly marked seed when every archived conformer fails stereo checks."""
    from crimm.IO.RTFParser import RTFParser
    from crimm.IO.PRMParser import categorize_lines, parse_line_dict
    from crimm.Modeller.TopoLoader import ParameterLoader, ResidueTopologySet
    from crimm.Modeller.MembraneBuilder import MembraneBuilder

    if len({part for part, _, _ in template["source_identity"]}) != 1:
        raise ValueError(f"No stereo-compliant reference conformer for compound {name}.")
    block = topology_text({name: template}, masses, template["chain_type"]).rsplit("END", 1)[0]
    for atoms, values in internal_coordinates[template["source_identity"][0][1]]:
        block += "IC " + " ".join(atoms) + " " + " ".join(map(str, values)) + "\n"
    definitions = ResidueTopologySet("lipid", rtf_parser=RTFParser(rtf_block=block + "END\n"))
    parameters = ParameterLoader()
    parameters.param_dict = parse_line_dict(categorize_lines(parameter_block.splitlines()))
    parameters.fill_ic(definitions, preserve=True)
    residue = definitions[name].create_residue()
    if residue is None or set(residue.child_dict) != {atom[0] for atom in template["atoms"]}:
        raise ValueError(f"The source IC table cannot construct a complete {name} seed.")
    names = [atom[0] for atom in template["atoms"]]
    coords = np.asarray([residue[atom].coord for atom in names])
    indices = {atom: index for index, atom in enumerate(names)}
    head = coords[[indices[atom] for atom in template["head_atoms"]]].mean(axis=0)
    tail = coords[[indices[atom] for atom in template["tail_atoms"]]].mean(axis=0)
    rotation = MembraneBuilder._rotation_matrix_from_vectors(head - tail, [0, 0, 1])
    coords = (coords - coords.mean(axis=0)) @ rotation.T
    if not valid_conformers(template, [coords]):
        raise ValueError(f"The source IC seed conflicts with the reference stereo state for {name}.")
    return coords


def components(psf):
    """Group covalently connected molecules, not individual sugar residues."""
    parents = list(range(len(psf.atoms)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for a, b in psf.bonds:
        parents[root(b - 1)] = root(a - 1)
    groups = defaultdict(list)
    for index in range(len(parents)):
        groups[root(index)].append(index)
    return list(groups.values())


def element(mass):
    name = min(ELEMENT_MASSES, key=lambda item: abs(mass - ELEMENT_MASSES[item]))
    if abs(mass - ELEMENT_MASSES[name]) > 0.05:
        raise ValueError(f"Unexpected membrane atom mass: {mass}.")
    return name


def molecule_name(atoms):
    residues = dict.fromkeys((atom.segid, atom.resid, atom.resname) for atom in atoms)
    names = tuple(sorted(name for _, _, name in residues))
    if len(names) == 1:
        return names[0], False
    if names not in COMPOUNDS:
        raise ValueError(f"Unknown compound lipid: {names}.")
    return COMPOUNDS[names], True


def canonical_template(atoms, indices, psf, compound):
    lookup = {source_index + 1: index for index, source_index in enumerate(indices)}
    if compound and len(atoms) > 4096:
        raise ValueError("Compound lipid exceeds the atom-name capacity.")
    names = [f"{element(atom.mass)}{index:03X}" if compound else atom.atomname
             for index, atom in enumerate(atoms)]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate lipid atom names.")
    source_residues = {(segid, resid): index for index, (segid, resid) in enumerate(
        dict.fromkeys((atom.segid, atom.resid) for atom in atoms))}
    template = {
        "atoms": [[name, atom.atomtype, atom.charge, atom.mass, element(atom.mass)]
                  for name, atom in zip(names, atoms)],
        "source_identity": [[source_residues[(atom.segid, atom.resid)], atom.resname, atom.atomname]
                            for atom in atoms],
        "groups": [lookup[start + 1] for start, _, _ in psf.groups if start + 1 in lookup],
    }
    for key in ("bonds", "angles", "dihedrals", "impropers", "cmap"):
        template[key] = [list(map(lookup.__getitem__, term)) for term in getattr(psf, key)
                         if term[0] in lookup]
    if psf.lonepairs:
        raise ValueError("Reference lipid lone pairs need explicit support.")
    return template


def compile_library(cache, catalogue, output):
    templates, conformers = {}, defaultdict(list)
    masses, doubles, parameter_blocks, internal_coordinates = {}, defaultdict(set), {}, {}
    for record in catalogue["membranes"]:
        code = record["id"]
        with tarfile.open(cache / f"{code}.tar.gz") as archive:
            for filename in ("step4_lipid.psf", "step4_lipid.crd", "membrane_restraint.str"):
                member = next(item for item in archive.getmembers() if Path(item.name).name == filename)
                if (cache / code / filename).read_bytes() != archive.extractfile(member).read():
                    raise ValueError(f"Cached {filename} differs from the source archive for {code}.")
        psf = read_psf(str(cache / code / "step4_lipid.psf"))
        atomtypes = {atom.atomtype for atom in psf.atoms if atom.resname != "TIP3"}
        source_masses, source_doubles, parameters, source_ics = reference_forcefield(cache / f"{code}.tar.gz", atomtypes)
        masses.update(source_masses)
        internal_coordinates[code] = source_ics
        for residue, pairs in source_doubles.items():
            doubles[residue].update(pairs)
        parameter_blocks[code] = "\n".join(parameters)
        entries = CRDParser(QUIET=True).create_namedtuples(str(cache / code / "step4_lipid.crd"))
        coords = np.asarray([entry.coord for entry in entries])
        counts = [Counter(), Counter()]
        signatures = {}
        for indices in components(psf):
            atoms = [psf.atoms[index] for index in indices]
            if {atom.resname for atom in atoms} == {"TIP3"}:
                continue
            name, compound = molecule_name(atoms)
            signature = tuple((atom.resname, atom.atomname, atom.atomtype, atom.charge, atom.mass) for atom in atoms)
            if name not in signatures:
                signatures[name] = signature
                template = canonical_template(atoms, indices, psf, compound)
                if name in templates and template != templates[name]:
                    raise ValueError(f"Reference templates differ for {name} in {code}.")
                templates[name] = template
            elif signature != signatures[name]:
                raise ValueError(f"Reference molecule identities differ for {name} in {code}.")
            xyz = coords[indices].copy()
            anchors = [i for i, atom in enumerate(atoms) if atom.atomname == "P"]
            anchor_z = xyz[anchors or list(range(len(xyz))), 2].mean()
            counts[0 if anchor_z > 0 else 1][name] += 1
            if anchor_z < 0:
                xyz[:, 1:] *= -1
            xyz -= xyz.mean(axis=0)
            conformers[name].append(xyz)
        aliases = ALIASES.get(code, {})
        expected = [{aliases.get(name, name): count for name, count in record[key].items()}
                    for key in ("upper_ratios", "lower_ratios")]
        actual = [dict(leaf) for leaf in counts]
        if actual != expected:
            corrected = [dict(leaf, MGDGPO=40) for leaf in expected]
            if code != "THYb" or actual != corrected:
                raise ValueError(f"Published counts differ from the reference for {code}: {actual} != {expected}.")
            record["source_discrepancies"] = [
                "The HTML table reports 45 MGDG-PO per leaflet and a total of 100, "
                "but its rows sum to 105. The reference PSF/CRD contains 40 MGDG-PO "
                "and 100 whole molecules per leaflet. The preset retains the reference inventory."
            ]
        record["published_upper_ratios"] = record["upper_ratios"]
        record["published_lower_ratios"] = record["lower_ratios"]
        record["upper_ratios"], record["lower_ratios"] = actual
        record["lipid_aliases"] = aliases
        record["whole_molecule_counts"] = actual
        if code == "aPM":
            restraint_text = (cache / code / "membrane_restraint.str").read_text()
            heights = {name: float(value) for name, value in re.findall(
                r"if\s+lipid\s+\.eq\.\s+(\S+)\s+then\s+calc\s+zpostop\s*=\s*([\d.]+)", restraint_text, re.I)}
            record["head_offsets_A"] = {name: heights[name] - heights["POPC"] for name in actual[0]}
            record["head_offsets_source"] = "membrane_restraint.str, relative to the POPC head plane"
            record["packing_warning"] = (
                "aPM reference-density packing remains experimental. "
                "Strict packing has not passed. Do not treat this preset as a validated preparation.")
        print(code, f"{sum(counts[0].values())}/{sum(counts[1].values())} whole molecules", flush=True)
    for name, template in templates.items():
        for _, atom_type, _, mass, _ in template["atoms"]:
            masses[atom_type] = mass
        first_record = next(record for record in catalogue["membranes"]
                            if name in record["upper_ratios"] or name in record["lower_ratios"])
        restraint_text = (cache / first_record["id"] / "membrane_restraint.str").read_text()
        placement_metadata(name, template, conformers[name][0], restraint_text)
        template["double_bonds"] = [
            [a, b] for a, b in template["bonds"]
            if template["source_identity"][a][0] == template["source_identity"][b][0]
            and frozenset([template["source_identity"][a][2], template["source_identity"][b][2]])
            in doubles[template["source_identity"][a][1]]
        ]
        stereo_guards(template, conformers[name], internal_coordinates[first_record["id"]])
        template["valid_conformers"] = valid_conformers(template, conformers[name])
        template["source_conformer_count"] = len(conformers[name])
        if not template["valid_conformers"]:
            coords = build_ic_conformer(name, template, masses, internal_coordinates[first_record["id"]],
                                        parameter_blocks[first_record["id"]])
            template["valid_conformers"] = [len(conformers[name])]
            template["generated_conformer"] = "source_IC"
            conformers[name].append(coords)
        print(name, f"{len(template['valid_conformers'])}/{len(conformers[name])} stereo-compliant conformers", flush=True)
    data = {"_templates": np.array(json.dumps(templates, separators=(",", ":")))}
    data["_lipid_rtf"] = np.array(topology_text(templates, masses, "Lipid"))
    data["_sterol_rtf"] = np.array(topology_text(templates, masses, "Sterol"))
    data.update({f"_parameters_{code}": np.array(block) for code, block in parameter_blocks.items()})
    for name, values in conformers.items():
        data[f"{name}_atoms"] = np.array([atom[0] for atom in templates[name]["atoms"]])
        data[f"{name}_coords"] = np.asarray(values)
    temporary = output.with_suffix(".tmp.npz")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **data)
    temporary.replace(output)
    return templates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalogue = json.loads(args.catalogue.read_text())
    for record in catalogue["membranes"]:
        actual = hashlib.sha256((args.cache / f"{record['id']}.tar.gz").read_bytes()).hexdigest()
        if actual != record["archive_sha256"]:
            raise ValueError(f"Reference archive hash differs for {record['id']}.")
    templates = compile_library(args.cache, catalogue, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(catalogue, indent=2, allow_nan=False) + "\n")
    print(f"Compiled {len(templates)} whole lipid types without waters.")


if __name__ == "__main__":
    main()
