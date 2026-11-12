"""Convert the complete lipid archive to lossless compact coordinates."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from crimm.Modeller.MembraneBuilder import LipidConformerLibrary


def pack_lipid_library(source, destination):
    library = LipidConformerLibrary(source)
    data = {
        "_source_sha256": np.array(hashlib.sha256(Path(source).read_bytes()).hexdigest()),
    }
    for resname, coords in library._coords_by_resname.items():
        for scale, dtype in ((100000, np.int32), (10000000000, np.int64)):
            encoded = np.rint(coords * scale)
            if (np.all(np.abs(encoded) < np.iinfo(dtype).max)
                    and np.array_equal(coords, encoded / scale)):
                encoded = encoded.astype(dtype)
                break
        else:
            encoded, scale = coords, 1
        data[f"{resname}_atoms"] = np.array(library._atoms_by_resname[resname])
        data[f"{resname}_coords"] = encoded
        data[f"{resname}_scale"] = np.array(scale)
        data[f"{resname}_members"] = np.array(library._members_by_resname[resname])
    np.savez_compressed(destination, **data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    pack_lipid_library(args.source, args.destination)
