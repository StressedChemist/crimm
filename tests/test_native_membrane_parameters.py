"""Check all named planar PPM3 systems and bounds-checked native failures."""

import importlib
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys

import numpy as np

from crimm.Modeller.MembraneOrienter import PPM3_MEMBRANE_TYPES
from test_native_membrane_orientation import ROOT, _run, native_build, reference


def test_native_membrane_parameter_table(tmp_path):
    source = tmp_path / "parameters.f90"
    source.write_text("""program check_parameters
      implicit none
      integer ppm_status, i, j
      common /ppm_error/ ppm_status
      character(3) codes(21)
      real matching(3)
      codes = [character(3) :: 'PMm','PMp','PMf','ERf','ERm','GOL','LYS','END','VAC', &
               'MOM','MIM','THp','THb','GnO','GnI','GpI','ARC','LPC','MPC','OPC','EPC']
      do i=1,21
        do j=1,2
          ppm_status=0
          call ppm3_parameters(codes(i),2,j,matching)
          if(ppm_status/=0) stop 1
          print '(a3, i2, 3f12.5)', codes(i), j, matching
        end do
      end do
      ppm_status=0
      call ppm3_parameters('MIC',1,1,matching)
      if(ppm_status/=1) stop 2
    end program
    """)
    library = Path(find_spec("crimm.Modeller._ppm3._native").origin)
    executable = tmp_path / "parameters"
    _run(["gfortran", str(source), str(library), f"-Wl,-rpath,{library.parent}", "-o", str(executable)])
    lines = _run([str(executable)]).stdout.splitlines()
    equilibrium = (32.0,30.6,30.6,28.7,28.7,30.0,30.7,30.0,30.4,22.7,28.4,
                   30.2,30.8,23.9,30.1,30.3,30.2,21.7,25.7,28.8,35.8)
    coefficients = (0.02,) * 9 + (0.01,) * 2 + (0.02,) * 2 + (0.015,) * 4 + (0.02,) * 4
    assert len(lines) == 42
    for index, (code, thickness, coefficient) in enumerate(zip(PPM3_MEMBRANE_TYPES, equilibrium, coefficients)):
        for beta in (False, True):
            fields = lines[2 * index + int(beta)].split()
            assert fields[0] == code
            match = 23.9 if beta and code not in {"MOM", "GnO"} else thickness
            stiffness = 0.01 if beta and code not in {"MOM", "GnO"} else coefficient
            np.testing.assert_allclose(list(map(float, fields[2:])), [thickness, match, stiffness], atol=1e-5)


def test_bounds_checked_native_handles_short_and_singular_geometry(reference, tmp_path):
    native = ROOT / "crimm/Modeller/_ppm3"
    library = tmp_path / "debug.so"
    sources = ["shim.f90", "driver.f", "prepare.f", "min.f", "rmsd.f", "solva.f",
               "watface.f", "tilting.f", "locate.f", "profile.f", "hbcor.f", "orient.f", "find_segm.f"]
    _run(["gfortran", "-O0", "-g", "-fPIC", "-shared", "-std=legacy", "-ffixed-line-length-none",
          "-fcheck=bounds", "-ffpe-trap=invalid,zero", *[str(native / name) for name in sources], "-o", str(library)], cwd=tmp_path)
    script = f"""
import ctypes, importlib, runpy
from pathlib import Path
import numpy as np
checks=runpy.run_path({str(ROOT / 'tests/test_native_membrane_orientation.py')!r})
module=importlib.import_module('crimm.Modeller.MembraneOrienter')
function=ctypes.CDLL({str(library)!r}).ppm3_orient
function.argtypes=module._native_function().argtypes
function.restype=None
model=checks['_model'](Path({str(reference)!r}))
chain=model.child_list[0]
for residue in list(chain)[3:]: chain.detach_child(residue.id)
coords,names,residues,chains,ids=module.MembraneOrienter(model)._protein_input()
rotation=np.empty((3,3),dtype=np.float64,order='F')
translation=np.empty(3,dtype=np.float64)
metrics=np.empty(6,dtype=np.float64)
matching=np.empty(3,dtype=np.float64)
path={str(native / 'res.lib').encode()!r}
baseline=None
for singular in (False,True,False):
    current=np.zeros_like(coords) if singular else coords.copy()
    status=ctypes.c_int()
    function(len(coords),current,names,residues,chains,ids,path,len(path),7.5,293.0,0,b'OPC',
             rotation,translation,metrics,matching,ctypes.byref(status))
    assert status.value in ((4,5,6) if singular else (0,)), status.value
    if not singular:
        assert np.isfinite(rotation).all() and np.isfinite(metrics).all()
        if baseline is None: baseline=(rotation.copy(),translation.copy(),metrics.copy())
        else:
            for current,expected in zip((rotation,translation,metrics),baseline):
                np.testing.assert_array_equal(current,expected)
print('DEBUG_NATIVE_RETURNED')
"""
    result = _run([sys.executable, "-c", script], cwd=ROOT, timeout=300)
    assert "DEBUG_NATIVE_RETURNED" in result.stdout
