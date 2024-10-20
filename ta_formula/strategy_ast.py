import argparse
import ast
import re

from .exceptions import PyxSyntaxError

cache_func_no_return_type_temp = """
    cdef int _{id}_defined = 0
    _{id} = None
    def {id}():
        nonlocal _{id}_defined, _{id}
        if _{id}_defined == 0:
            _{id} = {func}
            _{id}_defined = 1
        return _{id}
"""


def _get_subscript_value(node):
    if not isinstance(node, ast.Subscript):
        return node
    return _get_subscript_value(node.value)


def parse_pyx_file(pyx_file: str, params: dict, return_fileds: list, debug: bool):
    """
    第一行必须是cimport 开头
    必须有datas字段
    必须有ret字段
    只支持赋值语句和函数调用语句
    """
    with open(pyx_file, "r", encoding="utf8") as pyx:
        # cimport cannot parse by ast module
        code = ""
        while "cimport" in (line := pyx.readline()).strip():
            code += "\n"
        code += line
        code += pyx.read()

    try:
        st = ast.parse(code)
    except SyntaxError as e:
        raise PyxSyntaxError(f"{e.msg}, line {e.lineno}, {e.offset}") from e

    pyx_struct = {}

    temp_pyx_struct = {}
    for node in ast.iter_child_nodes(st):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            raise PyxSyntaxError(
                f"is not assignment, line {value.lineno}, {value.col_offset}"
            )
        if "targets" in node._fields:
            # ann = None
            target = node.targets[0]
        elif "target" in node._fields:
            # ann = ast.unparse(node.annotation)
            target = node.target
        if "id" not in target._fields:
            raise PyxSyntaxError(
                f"tuple unpacking not supported, line {value.lineno}, {value.col_offset}"
            )
        temp_pyx_struct[target.id] = node.value
        # temp_pyx_struct[target.id] = (ann, node.value)

    datas_struct = temp_pyx_struct.pop("datas", None)
    if not datas_struct:
        raise PyxSyntaxError(f'"datas" variable cannot be found.')
    pyx_struct["datas"] = ast.literal_eval(datas_struct)

    ret = temp_pyx_struct.pop("ret", None)
    if not ret:
        raise PyxSyntaxError(f'"ret" variable cannot be found.')
    # 选择性返回结果
    if return_fileds:
        ret_keys = []
        ret_values = []
        for ret_key, ret_value in zip(ret.keys, ret.values):
            for field in return_fileds:
                if field in ret_key.value:
                    ret_keys.append(ret_key)
                    ret_values.append(ret_value)
                    break
        ret = ast.Dict(ret_keys, ret_values)
    else:
        ret_keys = ret.keys
        ret_values = ret.values
    pyx_struct["ret"] = ast.unparse(ret)

    pyx_struct["constant_params"] = {}
    pyx_struct["datas_interface"] = []
    pyx_struct["indicators"] = {}
    for var_id, value in temp_pyx_struct.items():
        if isinstance(value, ast.Constant):
            pyx_struct["constant_params"][var_id] = value.value
        else:
            _value = _get_subscript_value(value)
            if isinstance(_value, ast.Name) and _value.id == "datas":
                # datas subscript
                pyx_struct["datas_interface"].append((var_id, ast.unparse(value)))
            elif isinstance(_value, ast.Call):
                pyx_struct["indicators"][var_id] = ast.unparse(value)
            else:
                raise PyxSyntaxError(
                    f"not support this defines, line {value.lineno}, {value.col_offset}"
                )

    # 更新params
    constant_params = pyx_struct["constant_params"]
    for k, v in params.items():
        if k in constant_params:
            constant_params[k] = v

    # 替换参数
    def replace1(match):
        return str(constant_params[match.group(0)])

    constant_params_replace_regex = "|".join(
        r"\b{}\b".format(k) for k in constant_params.keys()
    )
    # 1. 返回中的参数需要替换
    pyx_struct["ret"] = re.sub(
        constant_params_replace_regex, replace1, pyx_struct["ret"]
    )
    # 2. 函数中的参数需要替换
    for _id, func in pyx_struct["indicators"].items():
        pyx_struct["indicators"][_id] = re.sub(
            constant_params_replace_regex, replace1, func
        )
    # 替换函数
    indicator_ids = pyx_struct["indicators"].keys()

    def replace2(match):
        return f"{match.group(0)}()"

    pyx_struct["ret"] = re.sub(
        "|".join(r"\b{}\b".format(k) for k in indicator_ids),
        replace2,
        pyx_struct["ret"],
    )

    ret_key_strs = [k.value for k in ret_keys]
    data_params = [field for field, _ in pyx_struct["datas_interface"]]
    if debug:
        debug_str = str.join(
            ", ", [f"'debug_{field}': {field}" for field in data_params]
        )
        ret_key_strs += [f"debug_{field}" for field in data_params]
    cache_funcs = []
    for _id, func in pyx_struct["indicators"].items():
        cf = cache_func_no_return_type_temp.format(id=_id, func=func)
        cache_funcs.append(cf)
        if debug:
            ret_key_strs.append(f"debug_{_id}")
            debug_str += ", "
            debug_str += f"'debug_{_id}': {_id}()"
    if debug:
        pyx_struct["ret"] = pyx_struct["ret"][:-1] + ", " + debug_str + "}"
    ret_str = _dict_formatter(pyx_struct["ret"], ret_key_strs)
    code = f"""
# THIS IS AUTO GENERATED FILE, DO NOT MODIFY THIS FILE

#cython: language_level=3str
cimport ta_formula.indicators as ta


def calculate({str.join(', ', [field for field in data_params])}):
    {str.join('', cache_funcs)}

    return {ret_str}
    """
    return code, pyx_struct


key_value_pair = re.compile(r"[\'\"](\w+)[\'\"]\s*:(.*)")


def _dict_formatter(dict_str: str, keys):
    # 选择性返回结果
    s = [dict_str.index(k) - 1 for k in keys]
    rebuild_dict = "{\n"
    for i, j in zip(s, s[1:] + [-1]):
        m = re.match(key_value_pair, dict_str[i:j])
        k, v = m.group(1), m.group(2)
        rebuild_dict += f"        '{k}':{v}\n"
    rebuild_dict += "    }\n"
    return rebuild_dict


def tacompile_entry():
    parser = argparse.ArgumentParser(description="将pyx策略模板文件编译成策略文件")
    parser.add_argument("pyx_file", help="pyx策略模板")
    parser.add_argument("-d", "--debug", action="store_true", help="是否输出debug数据")
    args = parser.parse_args()
    code, _ = parse_pyx_file(args.pyx_file, {}, [], args.debug)
    output_file = args.pyx_file.replace(".pyx", "_out.pyx")
    with open(output_file, "w", encoding="utf8") as f:
        f.write(code)


if __name__ == "__main__":
    code, pyx_struct = parse_pyx_file("test_strategy.pyx", {}, ["open", "close"], False)
    print(code)
