import ast
import re

cache_func_with_return_type_temp = '''
    cdef int _{id}_defined = 0
    cdef {return_type} _{id}
    def {id}() -> {return_type}:
        nonlocal _{id}_defined, _{id}
        if _{id}_defined == 0:
            _{id} = {func}
            _{id}_defined = 1
        return _{id}
'''

cache_func_no_return_type_temp = '''
    cdef int _{id}_defined = 0
    _{id} = None
    def {id}():
        nonlocal _{id}_defined, _{id}
        if _{id}_defined == 0:
            _{id} = {func}
            _{id}_defined = 1
        return _{id}
'''

def parse_pyx_file(pyx_file: str, params: dict, return_fileds: list):
    '''
    必须有datas字段
    必须有ret字段
    只支持赋值语句和函数调用语句
    '''
    with open(pyx_file, 'r', encoding='utf8') as pyx:
        # cimport cannot parse by ast module
        code = ''
        while ('cimport' in (line:= pyx.readline()).strip()):
            code += '\n'
        code += line
        code += pyx.read()

    st = ast.parse(code)

    pyx_struct = {}

    temp_pyx_struct = {}
    for node in ast.iter_child_nodes(st):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            raise SyntaxError(f'line {node.lineno}, col_offset {node.col_offset} is not assignment.')
        if 'targets' in node._fields:
            ann = None
            target = node.targets[0]
        elif 'target' in node._fields:
            ann = ast.unparse(node.annotation)
            target = node.target
        if 'id' not in target._fields:
            raise SyntaxError(f'line {node.lineno}, col_offset {node.col_offset} tuple unpacking not supported.')
        temp_pyx_struct[target.id] = (ann, node.value)

    _, datas_struct = temp_pyx_struct.pop('datas', None)
    if not datas_struct:
        raise SyntaxError(f'"datas" variable cannot be found.')
    pyx_struct['datas'] = ast.literal_eval(datas_struct)

    _, ret = temp_pyx_struct.pop('ret', None)
    if not ret:
        raise SyntaxError(f'"ret" variable cannot be found.')
    # 选择性返回结果
    ret_keys = []
    ret_values = []
    for ret_key, ret_value in zip(ret.keys, ret.values):
        for field in return_fileds:
            if field in ret_key.value:
                ret_keys.append(ret_key)
                ret_values.append(ret_value)
                break
    ret = ast.Dict(ret_keys, ret_values)
    pyx_struct['ret'] = ast.unparse(ret)

    pyx_struct['constant_params'] = {}
    pyx_struct['datas_interface'] = []
    pyx_struct['indicators'] = {}
    for var_id, (ann, value) in temp_pyx_struct.items():
        if isinstance(value, ast.Constant):
            pyx_struct['constant_params'][var_id] = value.value
        elif isinstance(value, ast.Subscript):
            # datas subscript
            interface = (var_id, ast.unparse(value))
            if not interface[1].startswith('datas'):
                raise SyntaxError(f'line {value.lineno}, col_offset {value.col_offset} must be "datas" subscription')
            pyx_struct['datas_interface'].append(interface)
        elif isinstance(value, ast.Call):
            pyx_struct['indicators'][var_id] = (ann, ast.unparse(value))
        else:
            raise SyntaxError(f'line {value.lineno}, col_offset {value.col_offset} not support this defines')

    # 更新params
    constant_params = pyx_struct["constant_params"]
    for k, v in params.items():
        if k in constant_params:
            constant_params[k] = v
    # 替换参数
    def replace1(match):
        return str(constant_params[match.group(0)])
    pyx_struct["ret"] = re.sub('|'.join(r'\b{}\b'.format(k) for k in constant_params.keys()), replace1, pyx_struct["ret"])
    # 替换函数
    indicator_ids = pyx_struct["indicators"].keys()
    def replace2(match):
        return f'{match.group(0)}()'
    pyx_struct["ret"] = re.sub('|'.join(r'\b{}\b'.format(k) for k in indicator_ids), replace2, pyx_struct["ret"])

    data_params = [field for field,_ in pyx_struct['datas_interface']]
    cache_funcs = []
    for _id, (ann, func) in pyx_struct['indicators'].items():
        if ann is None:
            cf = cache_func_no_return_type_temp.format(id=_id, func=func)
        else:
            cf = cache_func_with_return_type_temp.format(id=_id, func=func, return_type=ann)
        cache_funcs.append(cf)
    ret_str= _dict_formatter(pyx_struct["ret"], ret_keys)
    code = f"""
# THIS IS AUTO GENERATED FILE, DO NOT MODIFY THIS FILE

#cython: language_level=3str
cimport numpy as np
cimport ta_formula._indicators as ta


def calculate({str.join(', ', ['np.ndarray '+field for field in data_params])}):
    {str.join('', cache_funcs)}

    return {ret_str}
    """
    return code, pyx_struct

key_value_pair = re.compile(r"[\'\"](\w+)[\'\"]\s*:(.*)")
def _dict_formatter(dict_str: str, keys):
    # 选择性返回结果
    s = [dict_str.index(k.value)-1 for k in keys]
    rebuild_dict = '{\n'
    for i, j in zip(s, s[1:]+[-1]):
        m = re.match(key_value_pair, dict_str[i:j])
        k, v = m.group(1), m.group(2)
        rebuild_dict += f"        '{k}':{v}\n"
    rebuild_dict += '    }\n'
    return rebuild_dict

if __name__ == '__main__':
    code, pyx_struct = parse_pyx_file('test_strategy.pyx', {}, ['open', 'close'])
    print(code)