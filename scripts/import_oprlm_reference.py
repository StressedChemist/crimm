"""Import the published OPRLM recipes and their CHARMM-GUI reference systems.

This is a data preparation command, not a simulation or a runtime download.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tarfile
import urllib.request


SOURCE_URL = "https://www.charmm-gui.org/?doc=archive&lib=biomembrane"
ARCHIVE_ROOT = "https://www.charmm-gui.org/archive/biomembrane"
PAPER_DOI = "10.1016/j.jmb.2025.168966"
RECIPE_DOI = "10.1021/acs.jcim.1c01514"
PPM_CODES = {
    "PMm": "PMm", "PMp": "PMp", "PMf": "PMf", "ERm": "ERm", "ERf": "ERf",
    "GOLm": "GOL", "GOLf": "GOL", "ENDm": "END", "LYSm": "LYS",
    "MOM": "MOM", "MIM": "MIM", "VAC": "VAC", "THYp": "THp", "THYb": "THb",
    "aPM": "ARC", "G-OM": "GnO", "G-IM": "GnI", "G+PM": "GpI",
}
PC_TYPES = ("POPC", "DOPC", "DSPC", "DMPC", "DPPC")


class RecipeParser(HTMLParser):
    """Read the named biomembrane tables, including unequal leaflet totals."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections = []
        self.section = None
        self.capture = None
        self.text = []
        self.row = None
        self.in_table = False

    def handle_starttag(self, tag, attrs):
        if tag == "h3":
            self.capture, self.text = "heading", []
        elif tag == "table" and self.section is not None:
            self.in_table = True
        elif tag == "tr" and self.in_table:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.capture, self.text = "cell", []
        elif tag == "li" and self.section is not None and not self.in_table:
            self.capture, self.text = "detail", []

    def handle_data(self, data):
        if self.capture is not None:
            self.text.append(data)

    def handle_endtag(self, tag):
        value = " ".join("".join(self.text).split())
        if tag == "h3" and self.capture == "heading":
            match = re.search(r"([A-Za-z+-]+)\s+\((.+)\)", value)
            self.section = None
            if match and match[1] in PPM_CODES:
                self.section = {"id": match[1], "name": match[2], "rows": [], "details": []}
                self.sections.append(self.section)
            self.capture = None
        elif tag in {"td", "th"} and self.capture == "cell":
            self.row.append(value)
            self.capture = None
        elif tag == "tr" and self.row is not None:
            self.section["rows"].append(self.row)
            self.row = None
        elif tag == "table":
            self.in_table = False
        elif tag == "li" and self.capture == "detail":
            self.section["details"].append(value)
            self.capture = None


def parse_recipes(text):
    parser = RecipeParser()
    parser.feed(text)
    sections = parser.sections
    if len(sections) != 18 or {item["id"] for item in sections} != set(PPM_CODES):
        raise ValueError("The source must contain exactly the 18 published biological recipes.")
    records = []
    for section in sections:
        upper, lower, areas, charges, descriptions = {}, {}, {}, {}, {}
        for row in section["rows"]:
            if len(row) not in (5, 7) or not row[2].isdigit() or row[0] == "TOTAL":
                continue
            name = "".join(row[0].split()).upper()
            if name == "CHOL":
                name = "CHL1"
            if name in descriptions:
                raise ValueError(f"Duplicate lipid {name} in {section['id']}.")
            descriptions[name] = row[1]
            upper[name] = int(row[2])
            lower[name] = int(row[3]) if len(row) == 7 else int(row[2])
            areas[name] = [None if entry == "N/A" else float(entry)
                           for entry in (row[4:6] if len(row) == 7 else [row[3], row[3]])]
            charges[name] = float(row[-1])
        totals = next(row for row in section["rows"] if row[0] == "TOTAL")
        reported_totals = [int(totals[2]), int(totals[3] if len(totals) == 7 else totals[2])]
        row_totals = [sum(upper.values()), sum(lower.values())]
        details = " ".join(section["details"])
        temperature = re.search(r"Temperature \(K\):\s*([\d.]+)", details)
        if temperature is None:
            raise ValueError(f"Missing temperature for {section['id']}.")
        dimensions = re.search(r"Size.*?:\s*([\d.]+) x ([\d.]+) x ([\d.]+)", details)
        if dimensions is None:
            raise ValueError(f"Missing reference box for {section['id']}.")
        code = section["id"]
        records.append({
            "id": code, "name": section["name"], "kind": "biological",
            "ppm_membrane": PPM_CODES[code], "temperature_K": float(temperature[1]),
            "reference_box_A": [float(value) for value in dimensions.groups()],
            "upper_ratios": {name: count for name, count in upper.items() if count},
            "lower_ratios": {name: count for name, count in lower.items() if count},
            "reference_areas_A2": areas, "reference_charges_e": charges,
            "lipid_descriptions": descriptions,
            "table_reported_totals": reported_totals, "table_row_totals": row_totals,
            "table_totals_match": reported_totals == row_totals,
            "archive_url": f"{ARCHIVE_ROOT}/{code}/{code}.tar.gz",
            "requires_calcium": code == "G-OM",
        })
    return records


def download(url, destination):
    """Cache a public reference download without replacing an existing file."""
    if not destination.is_file():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        with destination.open("xb") as handle:
            handle.write(data)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def reference_inventory(cache, records):
    """Read the actual molecular inventory before compiling lipid templates."""
    import numpy as np

    from crimm.IO.CRDParser import CRDParser
    from crimm.IO.PSFReader import read_psf

    for record in records:
        code = record["id"]
        destination = cache / code
        destination.mkdir(exist_ok=True)
        with tarfile.open(cache / f"{code}.tar.gz") as archive:
            for filename in ("step4_lipid.psf", "step4_lipid.crd", "membrane_restraint.str"):
                matches = [item for item in archive.getmembers()
                           if item.isfile() and Path(item.name).name == filename]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {filename} in {code}.")
                (destination / filename).write_bytes(archive.extractfile(matches[0]).read())
        psf = read_psf(str(destination / "step4_lipid.psf"))
        entries = CRDParser(QUIET=True).create_namedtuples(str(destination / "step4_lipid.crd"))
        expected = [(atom.segid, atom.resid, atom.resname, atom.atomname) for atom in psf.atoms]
        actual = [(atom.segid, str(atom.resid), atom.resname, atom.atomname) for atom in entries]
        if expected != actual:
            raise ValueError(f"PSF/CRD identities differ for {code}.")
        residues = defaultdict(list)
        for atom, entry in zip(psf.atoms, entries):
            residues[(atom.segid, atom.resid, atom.resname)].append(entry)
        counts = [Counter(), Counter()]
        for (_, _, resname), atoms in residues.items():
            anchor = [atom.coord[2] for atom in atoms if atom.atomname == "P"]
            if not anchor:
                anchor = [atom.coord[2] for atom in atoms]
            counts[0 if np.mean(anchor) > 0 else 1][resname] += 1
        crossing = [(a, b) for a, b in psf.bonds
                    if (psf.atoms[a - 1].segid, psf.atoms[a - 1].resid)
                    != (psf.atoms[b - 1].segid, psf.atoms[b - 1].resid)]
        record["reference_inventory"] = [dict(leaf) for leaf in counts]
        record["interresidue_bonds"] = len(crossing)
        record["lonepairs"] = len(psf.lonepairs)
        print(code, json.dumps(record["reference_inventory"]),
              f"interresidue bonds={len(crossing)}, lonepairs={len(psf.lonepairs)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-archives", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    page = args.cache / "biomembrane.html"
    page_hash = download(SOURCE_URL, page)
    records = parse_recipes(page.read_text())
    if args.download_archives:
        for record in records:
            record["archive_sha256"] = download(
                record["archive_url"], args.cache / f"{record['id']}.tar.gz")
            print(record["id"], record["archive_sha256"], flush=True)
    if args.audit:
        reference_inventory(args.cache, records)
    result = {
        "schema_version": 1, "paper_doi": PAPER_DOI, "recipe_doi": RECIPE_DOI,
        "recipe_url": SOURCE_URL, "recipe_sha256": page_hash,
        "membranes": records,
        "pc_cholesterol_families": list(PC_TYPES),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(records)} biological recipes and {len(PC_TYPES)} PC/cholesterol families.")


if __name__ == "__main__":
    main()
