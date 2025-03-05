import threading
import time

from .calculation import _CalculationCenter
from .dataflow_misc import RequestParser
from .strategy import get_strategy

__all__ = ["open_signal_stream"]

calculation_center = _CalculationCenter(threading.Lock())


def open_signal_stream(req_parser: RequestParser):
    strategy = get_strategy(*req_parser.get_strategy_params())
    # 准备所有需要的数据
    for backend, *args in req_parser.get_prepare_data_params(strategy):
        func = getattr(backend, "prepare")
        func(*args)  # 准备好这些数据字段，一直等待准备好为止
    # 生成计算单元
    units = []
    for one_unit_sources in req_parser.datasources:
        units.append(calculation_center.push(strategy, one_unit_sources))

    _waiter = DictEvent()
    for unit in units:
        unit._add_waiter(_waiter)
    try:
        while True:
            if _waiter.wait(timeout=0.1):
                signals = _waiter.clear()
                for signal in signals.values():
                    # 附加 发送时间
                    signal["calc_time"] = time.perf_counter_ns() - signal["calc_time"]
                    yield signal
    finally:
        for unit in units:
            unit._remove_waiter(_waiter)
            calculation_center.pop(unit)


class DictEvent:

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._value = {}

    def is_set(self):
        return bool(self._value)

    def add_result(self, unit_hash, result):
        with self._cond:
            self._value[unit_hash] = result
            self._cond.notify_all()

    def clear(self):
        old_value = self._value
        with self._cond:
            self._value = {}
        return old_value

    def wait(self, timeout=None):
        with self._cond:
            if self._value:
                return True
            return self._cond.wait(timeout)
