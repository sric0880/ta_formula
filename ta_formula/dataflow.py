import asyncio
import time
from collections import defaultdict
from concurrent import futures
from inspect import iscoroutinefunction

from .calculation import calculation_center
from .datasource import get_backend
from .strategy import get_strategy

__all__ = ['open_signal_stream']


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

    _waiter = futures._base._FirstCompletedWaiter()
    for unit in units:
        unit._add_waiter(_waiter)
    try:
        while True:
            if _waiter.event.wait(timeout=0.1):
                for fut in _waiter.finished_futures:
                    signal = fut.result()
                    # 附加 发送时间
                    signal['calc_time'] = time.perf_counter_ns() - signal['calc_time']
                    yield signal
                _waiter.finished_futures = []
                _waiter.event.clear()
    finally:
        for unit in units:
            unit._remove_waiter(_waiter)
            calculation_center.pop(unit)


def _prepare_arguments(datasources, datas_struct):
    ret = []
    for symbols, intervals in zip(zip(*datasources), datas_struct):
        symbols_by_db = defaultdict(set)
        for db, _symbol in symbols:
            symbols_by_db[db].add(_symbol)
        for db, _symbols_by_db in symbols_by_db.items():
            ret.append((db, '_prepare', list(_symbols_by_db), intervals))
    return ret


def _batch_call_backend_method(arguments):
    for backend, funcname, *args in arguments:
        func = getattr(backend, funcname)
        func(*args) # 准备好这些数据字段，一直等待准备好为止


async def _batch_call_backend_method_async(arguments):
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
        await asyncio.gather(async_funcs)


def _parse_datasources(datasources: list):
    '''
    parse like '[[{data_backend_clsname}.{bid}.{symbol}, ...], ...]'
    '''
    ret = []
    for dss in datasources:
        _dss = []
        for ds in dss:
            _, bid, symbol = ds.split('.')
            _dss.append((get_backend(bid), symbol))
        ret.append(_dss)
    return ret
