import os
import re

pxd_temp = """
#cython: language_level=3str

# THIS IS AUTO GENERATED FILE, DO NOT MODIFY THIS FILE
# USE `autogen_pyd.py` FOR UPDATING

cimport numpy as np

ctypedef np.double_t DTYPE_t
ctypedef (double, double) tuple_double2
ctypedef (double, double, double) tuple_double3
ctypedef (double, double, double, double) tuple_double4

#############################################
# START GENCODE FUNCTIONS FROM '{filename}'

{filecontent}

# END GENCODE FUNCTIONS FROM '{filename}'
#############################################


"""

header = """
#############################################
# START GENCODE FUNCTIONS FROM '{}'
"""

footer = """
# END GENCODE FUNCTIONS FROM '{}'
#############################################

"""

def find_includes(pyx_file):
    with open(pyx_file, 'r', encoding='utf8') as f:
        return re.findall(r'include "(.+)"', f.read())

def find_cdef_functions(file):
    all_functions = []
    with open(file, 'r', encoding='utf8') as f:
        for func in re.finditer(r'(^cdef .*):', f.read(), re.MULTILINE):
            s = func.group(1)
            if 'extern from' in s or 'check_' in s or ' _' in s or \
                'class ' in s:
                print(s, "不包含")
                continue
            all_functions.append(s)
    return all_functions

if __name__ == '__main__':
    module_name = 'ta_formula'
    pyx_file = '_indicators.pyx'
    copy_file = '_bool_func.pxi'
    except_files=['_ta_lib_common.pxi', '_unstable_periods.pxi', '_period_indicators.pxi']
    os.chdir(module_name)
    with open(copy_file, 'r', encoding='utf8') as f:
        pxd_temp = pxd_temp.format(filecontent=f.read(), filename=copy_file)
    all_includes = find_includes(pyx_file)
    for file in all_includes + [pyx_file]:
        if file in except_files:
            continue
        h = header.format(file)
        all_functions = find_cdef_functions(file)

        f = footer.format(file)
        pxd_temp += '\n'.join([h, *all_functions, f])
    with open('_indicators.pxd', 'w', encoding='utf8') as pxd:
        pxd.write(pxd_temp)


    