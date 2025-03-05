import asyncio
import time
from inspect import iscoroutinefunction

from .calculation import _CalculationCenter
from .dataflow_misc import RequestParser
from .strategy import get_strategy

__all__ = ["open_signal_stream"]

calculation_center = _CalculationCenter()


async def open_signal_stream(req_parser: RequestParser):
    # 这里编译pyx文件会阻塞线程，应该放到新开线程中去
    loop = asyncio.get_event_loop()
    strategy = await loop.run_in_executor(
        None, get_strategy, *req_parser.get_strategy_params()
    )
    # 准备所有需要的数据
    async_funcs = []
    for backend, *args in req_parser.get_prepare_data_params(strategy):
        func = getattr(backend, "prepare")
        if iscoroutinefunction(func):
            async_funcs.append(asyncio.create_task(func(*args)))
        else:
            async_funcs.append(loop.run_in_executor(None, func, *args))
    if async_funcs:
        await asyncio.gather(*async_funcs)
    # 生成计算单元
    units = []
    for one_unit_sources in req_parser.datasources:
        units.append(calculation_center.push(strategy, one_unit_sources))

    loop = asyncio.get_running_loop()
    _waiter = DictEvent(loop)
    for unit in units:
        unit._add_waiter(_waiter)
    try:
        while True:
            if await _waiter.wait():
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
    def __init__(self, loop):
        self.fut = None  # only one waiter
        self._value = {}
        self._loop = loop

    def is_set(self):
        return bool(self._value)

    def add_result(self, unit_hash, result):
        self._value[unit_hash] = result

        fut = self.fut
        if fut is not None:
            if not fut.done():
                fut.set_result(True)

    def clear(self):
        old_value = self._value
        self._value = {}
        return old_value

    async def wait(self):
        if self._value:
            return True

        self.fut = fut = self._loop.create_future()
        try:
            await fut
            return True
        finally:
            self.fut = None
