from pathlib import Path
import shutil
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class PPM3BuildExt(build_ext):
    def build_extension(self, extension):
        compiler = shutil.which("gfortran")
        if compiler is None:
            raise RuntimeError("PPM3 requires gfortran to build the native library.")

        library = Path(self.get_ext_fullpath(extension.name)).resolve()
        library.parent.mkdir(parents=True, exist_ok=True)
        build_dir = Path(self.build_temp).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [compiler, "-O2", "-fPIC", "-shared", "-std=legacy",
             "-ffixed-line-length-none", "-J", str(build_dir),
             *(str(Path(source).resolve()) for source in extension.sources),
             "-o", str(library)],
            cwd=build_dir,
            check=True,
        )


setup(
    ext_modules=[Extension(
        "crimm.Modeller._ppm3._native",
        sources=[str(Path("crimm/Modeller/_ppm3") / name) for name in (
            "shim.f90", "driver.f", "prepare.f", "min.f", "rmsd.f", "solva.f",
            "watface.f", "tilting.f", "locate.f", "profile.f", "hbcor.f",
            "orient.f", "find_segm.f",
        )],
    )],
    cmdclass={"build_ext": PPM3BuildExt},
)
