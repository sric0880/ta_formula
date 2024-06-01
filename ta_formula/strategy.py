import importlib
import logging
import os
import re
import tempfile
from distutils.sysconfig import get_python_lib
from itertools import chain

import numpy as np
from pyximport import pyximport

pyximport.install(build_in_temp=False, setup_args={'include_dirs': np.get_include()})

__all__ = ['Strategy', 'get_strategy']

pyx_strategy_temp = """
# THIS IS AUTO GENERATED FILE, DO NOT MODIFY THIS FILE

#cython: language_level=3str
cimport numpy as np
cimport ta_formula._indicators as ta

def calculate({datas_list}):
    return {ret_str}
"""

assignment = re.compile(r"(\w+)\s+=\s+(.*)")

number = re.compile(r"-?[\d\.]+")

key_value_pair = re.compile(r"[\'\"](\w+)[\'\"]\s*:(.*),")

_temp_pyx_folder = os.path.join(get_python_lib(), 'ta_formula_strategies')
os.makedirs(_temp_pyx_folder, mode=0o777, exist_ok=True)
package_init_py = os.path.join(_temp_pyx_folder, '__init__.py')
if not os.path.exists(package_init_py):
    with open(package_init_py, 'w') as f:
        f.write("")


def _get_strategy_hash(pyx_file, params, return_fileds):
    return hash(frozenset(chain((pyx_file,), params.items(), return_fileds)))

class Strategy:
    strategies_home_dir = '.' # 当pyx_file传入相对路径时，默认在此目录下查找

    def __init__(self, pyx_file: str, params: dict, return_fileds: list, _hash: int = 0):
        if _hash == 0:
            _hash = _get_strategy_hash(pyx_file, params, return_fileds)
        self._hash = _hash
        if not os.path.isabs(pyx_file):
            pyx_file = os.path.join(self.strategies_home_dir, pyx_file)
        self.pyx_filename = os.path.basename(pyx_file)
        pyx_struct = {}
        pyx_struct["scalar_params"] = {}
        pyx_struct["datas_params"] = {}
        pyx_struct["ret"] = ""
        with open(pyx_file, "r", encoding="utf8") as f:
            for l in f.readlines():
                l = l.strip()
                self._process_line(l, pyx_struct)
        # print(pyx_struct)

        # 更新params
        ret_str = pyx_struct["ret"]
        scalar_params = pyx_struct["scalar_params"]
        for k, v in params.items():
            if k in scalar_params:
                scalar_params[k] = v
        # 替换参数
        for k, v in scalar_params.items():
            ret_str = ret_str.replace(k, str(v), -1)

        datas_params_list = []
        self.datas_interface = []
        # 设置数据传参
        datas_params = pyx_struct["datas_params"]
        for k, v in datas_params.items():
            self.datas_interface.append((k, v))
            datas_params_list.append(f"np.ndarray {k}")
        self.datas_list = [None]*len(self.datas_interface)

        # 选择性返回结果
        dict_matchs = re.findall(key_value_pair, ret_str)
        rebuild_ret_dict = '{\n'
        for m in dict_matchs:
            k, v = m
            for field in return_fileds:
                if field in k:
                    rebuild_ret_dict += f"        '{k}':{v},\n"
        rebuild_ret_dict += '    }\n'

        self.pyx_strategy = pyx_strategy_temp.format(
            datas_list=", ".join(datas_params_list), ret_str=rebuild_ret_dict
        )

        # compile
        with tempfile.NamedTemporaryFile('w', suffix=self.pyx_filename, encoding='utf8', dir=_temp_pyx_folder, delete=False) as temp_pyx:
            temp_pyx.write(self.pyx_strategy)
        try:
            modulename = os.path.basename(temp_pyx.name)
            old_level = logging.getLogger().level
            # BUG: pyximport will change the level of root logger
            calculator = importlib.import_module('ta_formula_strategies.'+modulename.replace('.pyx', ''))
            logging.getLogger().setLevel(old_level)
            self.calculate = lambda: calculator.calculate(*self.datas_list)
            self.calculate_x = lambda x: calculator.calculate(*x)
        finally:
            try:
                os.remove(temp_pyx.name)
                os.remove(temp_pyx.name.replace('.pyx', '.c'))
            except:
                pass

    def __repr__(self) -> str:
        return self.pyx_strategy

    def __str__(self) -> str:
        return f'{self.pyx_filename}_{self._hash}'
    
    def feed_datas(self, datas):
        ''' 数据保存在strategy中，直接调用`calculate` '''
        for i, (k, v) in enumerate(self.datas_interface):
            self.datas_list[i] = eval(v, {}, {"datas": datas})

    def feed_external_datas(self, datas, datas_list):
        ''' 数据保存在外部的`datas`中，调用`calculate_x`将`datas`传入，用于策略共享 '''
        for i, (k, v) in enumerate(self.datas_interface):
            datas_list[i] = eval(v, {}, {"datas": datas})

    def _process_line(self, l: str, pyx_struct):
        if not l:
            return
        if "cimport" in l:
            return
        m = re.fullmatch(assignment, l)
        if m:
            var = m.group(1)
            value = m.group(2)
            if var == "datas":
                if hasattr(self, 'datas'):
                    raise RuntimeError('datas has defined again in pyx file')
                self.datas_struct = eval(value)
            elif var == "ret":
                pyx_struct["ret"] += value
            else:
                mm = re.fullmatch(number, value)
                if mm:
                    pyx_struct["scalar_params"][var] = (
                        float(value) if "." in value else int(value)
                    )
                else:
                    pyx_struct["datas_params"][var] = value
        else:
            pyx_struct["ret"] += l + "\n"

strategy_center = {}

def get_strategy(pyx_file: str, params: dict, return_fileds: list):
    # 优先从策略中心取
    _hash = _get_strategy_hash(pyx_file, params, return_fileds)
    strategy = strategy_center.get(_hash, None)
    if strategy is None:
        strategy_center[_hash] = strategy = Strategy(pyx_file, params, return_fileds, _hash)
    return strategy
