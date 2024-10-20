import os
import re

pxd_header = """
#cython: language_level=3str

# THIS IS AUTO GENERATED FILE, DO NOT MODIFY THIS FILE
# USE `autogen_pyd.py` FOR UPDATING

cimport numpy as np

ctypedef (double, double) tuple_double2
ctypedef (double, double, double) tuple_double3
ctypedef (double, double, double, double) tuple_double4

ctypedef fused numeric_dtype:
    int
    double
    long long

"""

copy_content = """
#############################################
# START FUNCTIONS COPY FROM '{filename}'
{filecontent}
# END FUNCTIONS COPY FROM '{filename}'
#############################################


"""

module_header = """
#############################################
# START FUNCTIONS GEN FROM '{filename}'
"""

module_footer = """
# END FUNCTIONS GEN FROM '{filename}'
#############################################
"""


def find_includes(pyx_files):
    ret = set()
    for pyx_file in pyx_files:
        with open(pyx_file, "r", encoding="utf8") as f:
            for file in re.findall(r'include "(.+)"', f.read()):
                ret.add(file)
    return list(ret)


def find_cdef_functions(file):
    all_functions = []
    with open(file, "r", encoding="utf8") as f:
        for func in re.finditer(r"(^cdef .*):", f.read(), re.MULTILINE):
            s = func.group(1)
            if "extern from" in s or "check_" in s or " _" in s or "class " in s:
                print(s, "不包含")
                continue
            all_functions.append(s)
    return all_functions


if __name__ == "__main__":
    module_name = "ta_formula"
    pyx_files = ["indicators.pyx"]
    copy_files = ["_bool_func.pxi"]
    except_files = [
        "_ta_lib_common.pxi",
        "_unstable_periods.pxi",
        "_period_indicators.pxi",
    ]
    os.chdir(module_name)
    gen_codes = [pxd_header]
    for copy_file in copy_files:
        print("copy file", copy_file)
        with open(copy_file, "r", encoding="utf8") as f:
            gen_codes.append(
                copy_content.format(filecontent=f.read(), filename=copy_file)
            )
    all_includes = find_includes(pyx_files)
    print("find includes: ", all_includes)
    for file in all_includes + pyx_files:
        if file in except_files:
            continue
        print("append file", file)
        gen_codes.append(module_header.format(filename=file))
        for function in find_cdef_functions(file):
            gen_codes.append(function)
        gen_codes.append(module_footer.format(filename=file))

    with open("indicators.pxd", "w", encoding="utf8") as pxd:
        pxd.write("\n".join(gen_codes))
