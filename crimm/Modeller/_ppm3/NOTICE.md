# PPM3 native protein positioning

CRIMM uses PPM code obtained from Andrei Lomize and Irina Pogozheva, the main authors.

Source: https://cggit.cc.lehigh.edu/biomembhub/ppm3_server_code.git
Revision: `5784f088032e5b9fb35147815f67500b8cc94855`.

## License

PPM3 is distributed under the GNU General Public License, version 3, on the
same terms as PPM2. On 2026-11-12, the CRIMM maintainer reported confirmation
from the PPM3 authors that these terms apply to this integration, including
the modified native source, compiled library, supporting files, and original
source reference archive. This notice records that maintainer confirmation.

A copy of the license is included in [LICENSE](LICENSE). Preserve attribution
and modification notices, and provide corresponding source when distributing
compiled versions as required by the GPL. CRIMM's own code retains its
GPL-3.0-or-later license.

## Scope and source changes

One planar membrane is supported. Curvature, micelles, multiple membranes,
file output, and report-only routines are omitted. The 21 named bilayer systems
retain upstream thickness and mismatch stiffness. `OPC` is the default. The
beta-structure override remains active. The profile is symmetric DOPC. Explicit
lipid mixtures do not change this profile. Native arithmetic uses default REAL.
The C ABI uses doubles.

Modified for CRIMM on 2026-10-12:

- `shim.f90` adds ISO_C_BINDING, finite checks, error status, and invalid-arithmetic
  and division-by-zero checks. It restores these flags to their caller values.
- `prepare.f` adapts residue preparation to arrays and an explicit library path.
  pH and temperature are explicit, with defaults 7.5 and 293 K. Temperature
  affects ionization preparation, not the fixed profile constants.
- `driver.f` follows planar PPM3 through preparation, orientation, segment
  detection, bundle axis, water-facing atoms, optimization, final `locate`, and
  projection. The final `locate` is essential because the thickness scan does
  not transform coordinates. Energy includes quadratic hydrophobic mismatch.
- The rigid fit uses the first ten scored atoms. The Python adapter puts the
  first available backbone N first. Topology uses this N, not a cap atom or an
  arbitrary first file atom. `in` gives negative Z and `out` gives positive Z.
  The topology flip is a proper 180-degree X rotation.
- Reachable STOP statements return status. COMMON state requires serial calls.
  The Python adapter uses a lock. SASA bounds reset on each call.
- PPM2 adapter safety corrections were retained where the planar routines match
  PPM3. Internal capacities and planar search grids match PPM3. Public limits
  remain 150,000 atoms, 20,000 residues, and 60 chains.
- CA windows use `nca`, not undefined `nres`. Short-chain windows stay in bounds.
  Diagnostic CA indices use windows, not author IDs. `ichange` starts at zero.
- Dummy arrays use assumed sizes. Grids are checked before integer multiplication.
  Slice and profile indices stay in bounds. Bundle-axis normalization rejects
  zero axes and handles an axis already parallel to Z. Inverse-cosine inputs
  are bounded. Projection initializes its default rotation. Missing favorable
  insertion minima return an error.

The adapter accepts explicit neutral HSE through native HIS typing. It rejects
HSD, HSP, unresolved HIS, and unsupported patches. It does not change chemistry.
Caps and cofactors receive the transform but do not score.

## ABI

```text
ppm3_orient(n, xyz, atom_names, residue_names, chain_ids, residue_ids,
            reslib_path, path_length, pH, temperature, topology, system_code,
            rotation, translation, metrics, matching, status)
```

Scalar inputs use pass-by-value. `xyz` is a Fortran 3-by-n double array. Atom and
residue names are four space-padded ASCII bytes. Each chain ID is one ASCII
byte. Residue IDs are C integers. The library path uses an explicit length.
The system code is three case-sensitive ASCII bytes. Topology is 0 for `in`
or 1 for `out`.

The transform is `rotation * input + translation`. The matrix is Fortran 3-by-3.
`matching` contains nominal equilibrium thickness, effective matching thickness,
and mismatch coefficient. Thicknesses are in angstroms.

The six metrics are transfer energy, full hydrophobic thickness, optimizer tilt,
positioning spread, tilt spread, and initial model number. Positioning spread is
Z-shift RMS for model 1 and thickness-range half-width for models 2 and 3. The
spreads are not confidence intervals. Printed dummy thickness and later
inertia-axis tilt are different report values.

Status: 0 success, 1 invalid input, 2 residue-library error, 3 unsupported typing,
4 capacity exceeded, 5 no favorable minimum, 6 undefined geometry or arithmetic.
Nonzero status invalidates every output.
