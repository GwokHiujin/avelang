#!/usr/bin/env python3
"""
Setup script for ave-lang with CMake-based C++ extensions.
"""

import os
import shutil
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Extension that uses CMake to build."""

    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Custom build_ext command that uses CMake."""

    @staticmethod
    def _working_ninja():
        """Return the path to a working ninja executable in the current env."""
        ninja = shutil.which("ninja")
        if not ninja:
            return None

        try:
            subprocess.run(
                [ninja, "--version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            return None

        return ninja

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DWITH_PYTHON=ON",
            f"-DAVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY={extdir}",
        ]

        ninja = self._working_ninja()
        if ninja:
            cmake_args.extend(["-GNinja", f"-DCMAKE_MAKE_PROGRAM={ninja}"])

        if "CMAKE_ARGS" in os.environ:
            cmake_args.extend(os.environ["CMAKE_ARGS"].split())

        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        # Set environment variables for cmake
        env = os.environ.copy()
        env["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_temp, env=env)


setup(
    ext_modules=[CMakeExtension("_avelang_bindings")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
