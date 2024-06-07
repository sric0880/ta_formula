import asyncio
import time
from inspect import iscoroutinefunction

from .calculation import _CalculationCenter
from .dataflow_misc import _parse_datasources, _prepare_arguments
from .strategy import get_strategy

__all__ = ['open_signal_stream']

calculation_center = _CalculationCenter()

async def open_signal_stream(request):
    datasources = request['datasources']
    datasources = _parse_datasources(datasources)
    # TODO: 这里编译pyx文件会阻塞线程，应该放到新开线程中去
    strategy = get_strategy(request['pyx_file'], request['params'], request['return_fields'])
    datas_struct = request['datas'] if 'datas' in request else strategy.datas_struct
    # 准备所有需要的数据
    call_prepare_args = _prepare_arguments(datasources, datas_struct)
    await _batch_call_backend_method(call_prepare_args)
    units = []
    for dss in datasources:
        symbol_infos = []
        for (db, symbol), intervals in zip(dss, datas_struct):
            symbol_infos.append((db, symbol, intervals))
        units.append(calculation_center.push(strategy, symbol_infos))

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
                    signal['calc_time'] = time.perf_counter_ns() - signal['calc_time']
                    yield signal
    finally:
        for unit in units:
            unit._remove_waiter(_waiter)
            calculation_center.pop(unit)


async def _batch_call_backend_method(arguments):
    async_funcs = []
    for backend, funcname, *args in arguments:
        func = getattr(backend, funcname)
        if iscoroutinefunction(func):
            async_funcs.append(asyncio.create_task(func(*args)))
        else:
            async_funcs.append(
                asyncio.get_event_loop().run_in_executor(None, func, *args)
            )
    if async_funcs:
        await asyncio.gather(*async_funcs)


class DictEvent:
    def __init__(self, loop):
        self.fut = None # only one waiter
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
