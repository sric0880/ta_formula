import threading
import time

from .calculation import _CalculationCenter
from .dataflow_misc import _parse_datasources, _prepare_arguments
from .strategy import get_strategy

__all__ = ['open_signal_stream']

calculation_center = _CalculationCenter(threading.Lock())

def open_signal_stream(request):
    datasources = request['datasources']
    strategy = get_strategy(request['pyx_file'], request['params'], request['return_fields'])
    datas_struct = request['datas'] if 'datas' in request else strategy.datas_struct
    datasources = _parse_datasources(datasources)
    # 准备所有需要的数据
    call_prepare_args = _prepare_arguments(datasources, datas_struct)
    _batch_call_backend_method(call_prepare_args)
    units = []
    for dss in datasources:
        symbol_infos = []
        for (db, symbol), intervals in zip(dss, datas_struct):
            symbol_infos.append((db, symbol, intervals))
        units.append(calculation_center.push(strategy, symbol_infos))

    _waiter = DictEvent()
    for unit in units:
        unit._add_waiter(_waiter)
    try:
        while True:
            if _waiter.wait(timeout=0.1):
                signals = _waiter.clear()
                for signal in signals.values():
                    # 附加 发送时间
                    signal['calc_time'] = time.perf_counter_ns() - signal['calc_time']
                    yield signal
    finally:
        for unit in units:
            unit._remove_waiter(_waiter)
            calculation_center.pop(unit)


def _batch_call_backend_method(arguments):
    for backend, funcname, *args in arguments:
        func = getattr(backend, funcname)
        func(*args) # 准备好这些数据字段，一直等待准备好为止


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
