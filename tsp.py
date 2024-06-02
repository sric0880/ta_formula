import ast


with open('test_strategy.pyx', 'r') as pyx:
    code = pyx.read()

st = ast.parse(code)
# print(st)

# for node in ast.walk(st):
#     print(node)
# 必须有datas字段
# 必须有ret字段
# 只支持赋值语句和函数调用语句
pyx_struct = {}

temp_pyx_struct = {}
for node in ast.iter_child_nodes(st):
    if not isinstance(node, ast.Assign):
        raise SyntaxError(f'line {node.lineno}, col_offset {node.col_offset} is not assignment.')
    var = node.targets[0]
    if 'id' not in var._fields:
        raise SyntaxError(f'line {node.lineno}, col_offset {node.col_offset} tuple unpacking not supported.')
    varvalue = node.value
    temp_pyx_struct[var.id] = node.value
    # temp_pyx_struct[node]

datas_struct = temp_pyx_struct.get('datas', None)
if not datas_struct:
    raise SyntaxError(f'"datas" variable cannot be found.')
temp_pyx_struct['datas'] = ast.literal_eval(datas_struct)

ret = temp_pyx_struct.get('ret', None)
if not ret:
    raise SyntaxError(f'"ret" variable cannot be found.')
temp_pyx_struct['ret'] = ast.unparse(ret)

# temp_pyx_struct['CLOSE'] = ast.literal_eval(temp_pyx_struct['CLOSE'])

print(temp_pyx_struct)