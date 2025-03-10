import importlib
import logging
import os
import tempfile
from distutils.sysconfig import get_python_lib
from itertools import chain

import numpy as np
from pyximport import pyximport

from .exceptions import DatasListNotMatch
from .strategy_ast import parse_pyx_file

pyximport.install(build_in_temp=False, setup_args={"include_dirs": np.get_include()})

__all__ = ["Strategy", "get_strategy"]

_temp_pyx_folder = os.path.join(get_python_lib(), "ta_formula_strategies")
os.makedirs(_temp_pyx_folder, mode=0o777, exist_ok=True)
package_init_py = os.path.join(_temp_pyx_folder, "__init__.py")
if not os.path.exists(package_init_py):
    with open(package_init_py, "w") as f:
        f.write("")


def _get_strategy_hash(pyx_file, params, return_fileds):
    return hash(frozenset(chain((pyx_file,), params.items(), return_fileds)))


class Strategy:
    strategies_home_dir = "."  # 当pyx_file传入相对路径时，默认在此目录下查找

    def __init__(
        self,
        pyx_file: str,
        params: dict,
        return_fileds: list,
        _hash: int = 0,
        debug=False,
    ):
        if _hash == 0:
            _hash = _get_strategy_hash(pyx_file, params, return_fileds)
        self._hash = _hash
        if not os.path.isabs(pyx_file):
            pyx_file = os.path.join(self.strategies_home_dir, pyx_file)
        self.pyx_filename = os.path.basename(pyx_file)

        self.pyx_code, pyx_struct = parse_pyx_file(
            pyx_file, params, return_fileds, debug
        )
        self.datas_interface = pyx_struct["datas_interface"]
        self.datas_struct = pyx_struct["datas"]
        self._datas_list = [None] * len(self.datas_interface)

        # compile
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=self.pyx_filename,
            encoding="utf8",
            dir=_temp_pyx_folder,
            delete=False,
        ) as temp_pyx:
            temp_pyx.write(self.pyx_code)
        temp_pyx_bld = temp_pyx.name + "bld"
        with open(temp_pyx_bld, "w", encoding="utf8") as build_file:
            build_file.write(
                """
from distutils.extension import Extension
def make_ext(name, filename):
    return Extension(name=name, sources=[filename],define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
"""
            )
        try:
            modulename = os.path.basename(temp_pyx.name)
            old_level = logging.getLogger().level
            # BUG: pyximport will change the level of root logger
            calculator = importlib.import_module(
                "ta_formula_strategies." + modulename.replace(".pyx", "")
            )
            logging.getLogger().setLevel(old_level)
            self.calculate = lambda: calculator.calculate(*self._datas_list)
            self.calculate_x = lambda x: calculator.calculate(*x)
        finally:
            try:
                os.remove(temp_pyx.name)
                os.remove(temp_pyx.name.replace(".pyx", ".c"))
                os.remove(temp_pyx_bld)
            except:
                pass

    def __repr__(self) -> str:
        return self.pyx_code

    def __str__(self) -> str:
        return f"{self.pyx_filename}_{self._hash}"

    def feed_datas(self, datas):
        """数据保存在strategy中，直接调用`calculate`"""
        try:
            for i, (k, v) in enumerate(self.datas_interface):
                self._datas_list[i] = eval(v, {}, {"datas": datas})
        except IndexError as e:
            raise DatasListNotMatch(f"{v} index out of range") from e

    def feed_external_datas(self, datas, datas_list):
        """数据保存在外部的`datas`中，调用`calculate_x`将`datas`传入，用于策略共享"""
        try:
            for i, (k, v) in enumerate(self.datas_interface):
                datas_list[i] = eval(v, {}, {"datas": datas})
        except IndexError as e:
            raise DatasListNotMatch(f"{v} index out of range") from e


strategy_center = {}


def get_strategy(pyx_file: str, params: dict, return_fileds: list):
    # 优先从策略中心取
    _hash = _get_strategy_hash(pyx_file, params, return_fileds)
    strategy = strategy_center.get(_hash, None)
    if strategy is None:
        strategy_center[_hash] = strategy = Strategy(
            pyx_file, params, return_fileds, _hash
        )
    return strategy
