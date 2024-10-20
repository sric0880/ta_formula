import sys

import numpy
from setuptools import Extension, setup

lib_talib_name = "ta_lib"  # the underlying C library's name

if any(s in sys.platform for s in ["darwin", "linux", "bsd", "sunos"]):
    include_dirs = [
        "/usr/include",
        "/usr/local/include",
        "/opt/include",
        "/opt/local/include",
        "/opt/homebrew/include",
        "/opt/homebrew/opt/ta-lib/include",
    ]
    library_dirs = [
        "/usr/lib",
        "/usr/local/lib",
        "/usr/lib64",
        "/usr/local/lib64",
        "/opt/lib",
        "/opt/local/lib",
        "/opt/homebrew/lib",
        "/opt/homebrew/opt/ta-lib/lib",
    ]

elif sys.platform == "win32":
    lib_talib_name = "ta_libc_cdr"
    include_dirs = [
        r"c:\ta-lib\c\include"
    ]  # copy all .h files to c:\ta-lib\c\include\ta-lib\ directory
    library_dirs = [r"C:\ProgramData\Anaconda3\Library\lib"]  # change to your lib path

include_dirs.append(numpy.get_include())

ext_modules = [
    Extension(
        "ta_formula.indicators",
        ["ta_formula/indicators.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=[lib_talib_name],
        runtime_library_dirs=[] if sys.platform == "win32" else library_dirs,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(ext_modules=ext_modules)
