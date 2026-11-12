# PPM3 reference fixture

Source: https://cggit.cc.lehigh.edu/biomembhub/ppm3_server_code.git
Revision: `5784f088032e5b9fb35147815f67500b8cc94855`.
Archive SHA256: `ca599b95e7ec977abe9bee778cd6444daf2d89dbffd6ed537e4da9d0d3f22f46`.

The archive contains original source bytes from `git archive`, not CRIMM's array
driver or patched routines. The PPM3 authors' approval to use the same GNU GPL
version 3 terms as PPM2 was confirmed by the CRIMM maintainer on 2026-11-12.
See the [component notice](../../crimm/Modeller/_ppm3/NOTICE.md) and
[license](../../crimm/Modeller/_ppm3/LICENSE). The archive bytes and checksum
above are unchanged. The offline PDB fixtures come from the separately
documented [PPM2 archive](ppm2_reference.md).

The tests compile the original main program separately. Reference corrections
replace undefined `nres` with `nca` in CA windows, initialize `ichange`, initialize
the default projection rotation, and initialize `nlipid`.
Inverse-cosine arguments in `watface` and `hbcor` are bounded to correct rounding
outside [-1, 1].
Atom count and six metrics are printed after final planar positioning. Input type 2 selects one
membrane and `planar`. Upstream type 1 enables curvature.

Atoms use the adapter's canonical order, with backbone N first. Coordinates map
back to the original model order. Coordinate tolerance is 0.002 angstroms. Metric
tolerances are 2e-5 absolute and 2e-6 relative. The reference is not linked to
CRIMM's driver. This does not claim parity with undefined original behavior.
