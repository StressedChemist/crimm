# PPM2 reference fixture

The archive contains source files, `1rsy.pdb`, and `1gzm.pdb` from the PPM2 repository:

- Source: https://cggit.cc.lehigh.edu/biomembhub/ppm2_server_code.git
- Revision: `0de704acd10fbf7f4fbafb9bc925e01b06f38ceb`
- Archive SHA256: `df4976ac75de5228e74fc6d4409c6c3556cf3200a672824a3363daf2fe03a7a7`
- `opm.f` SHA256: `3c444a47782f8dd24008e8c13fdccf19d9750aa517a2d2172de97aae1e0a53ea`
- `find_segm.f` SHA256: `dfbf0d20cc48e926ea077c891ffd45a655809b76a2d3d0baa1dd259d55ebf931`
- `1rsy.pdb` SHA256: `7c75f8b846d0bd12c36eb44c8be46ac559da27fed9d7f94bb15e53dbd45fcc58`
- `1gzm.pdb` SHA256: `84c75243df607759a2e200285be04bae9462f5a167c90ab641edefe99416e0eb`

The archive preserves the original bytes. It includes the original GPLv3 license.
The current PPM3 tests and CHARMM example read the two PDB structures from this archive.
They do not compile the PPM2 driver. Native reference tests use the separate [PPM3 archive](ppm3_reference.md).

The coordinate fixture uses the complete protein fragment from residues 193 through 264 of `1rsy.pdb`.
The test does not reconstruct the incomplete residues 189 through 192.
The CRIMM fixture uses the explicit HSE label for the three native HIS residues.
The original driver receives the native HIS label and the same heavy-atom coordinates.
The test compares the final rigid transform, including the native projection and the topology flip.

The transmembrane fixture uses complete residues 1 through 326 from `1gzm.pdb`.
Its 2,592 heavy atoms exercise the native bundle-axis branch and model 2 classification.
The comparison uses the same six metrics and every transformed coordinate.
Coordinate agreement requires an absolute tolerance of 0.002 Å.
Metric agreement uses absolute and relative tolerances of `2e-5` and `2e-6`.
