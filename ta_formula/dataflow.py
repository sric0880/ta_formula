import asyncio
import logging
import threading
from collections import defaultdict
from concurrent.futures import Future, as_completed
from contextlib import asynccontextmanager, contextmanager
from inspect import iscoroutinefunction

from .datasource import get_backend
from .strategy import get_strategy

__all__ = ['open_dataflow_async', 'open_dataflow']

class _CalculationCenter:
    def __init__(self) -> None:
        self.calc_units = {}

    def push(self, strategy, symbol_infos):
        _hash = hash((strategy._hash, tuple((db.bid, symbol, *intervals) for db, symbol, intervals in symbol_infos)))
        unit = self.calc_units.get(_hash, None)
        if unit is None:
            self.calc_units[_hash] = unit = _CalculateUnit(strategy, symbol_infos)
            unit._hash = _hash
        return unit

    def signal_from(self, units):
        datas_ready = threading.Event()
        for unit in units:
            unit.register_event(datas_ready)
        while True:
            if datas_ready.wait(timeout=0.1):
                for unit in units:
                    if unit.datas_ready:
                        yield unit.calculate()

    def pop(self):
        pass

calculation_center = _CalculationCenter()


class _CalculateUnit:
    def __init__(self, strategy, symbol_infos) -> None:
        self.events = []
        self.datas_ready = False
        self.cached_result = None
        self.strategy = strategy
        self.calc_lock = threading.Lock()

    def register_event(self, event):
        self.events.append(event)

    def calculate(self):
        with self.calc_lock:
            if self.datas_ready:
                self.cached_result = self.strategy.calculate()


def open_dataflow(request):
    datasources = request['datasources']
    strategy = get_strategy(request['strategy_name'], request['params'], request['return_fields'])
    datas_struct = request['datas'] if 'datas' in request else strategy.datas_struct
    datasources = _parse_datasources(datasources)
    call_prepare_args = _prepare_arguments(datasources, strategy)
    _batch_call_backend_method(call_prepare_args) # 准备所有需要的数据
    units = []
    for dss in datasources:
        symbol_infos = []
        for (db, symbol), intervals in zip(dss, datas_struct):
            symbol_infos.append((db, symbol, intervals))
        units.append(calculation_center.push(strategy, symbol_infos))

    try:
        yield from calculation_center.signal_from(units)
    except:
        pass
    finally:
        calculation_center.pop(units)

        # datas_list = []
        # datas_futures_list = []
        # all_futures = set()
        # for dss in datasources:
        #     datas = []
        #     datas_futures = []
        #     for (db, symbol), intervals in zip(dss, strategy.datas_struct):
        #         datas.append([db.all_datas[symbol][interval] for interval in intervals])
        #         futs = [db._futures[symbol][interval] for interval in intervals]
        #         datas_futures.append(futs)
        #         # all_futures+=futs
        #         all_futures.update(*futs)
        #     datas_list.append(datas)
        #     datas_futures_list.append(datas_futures)

        # all_futures = list(all_futures)
        # all_datas_list = list(zip(datas_list, datas_futures_list))

        # calculation_tasks = []

        # def _signal_generator():


        # strategy.feed_datas(datas)
        # strategy.calculate()

        # yield _signal_generator()

def _prepare_arguments(datasources, strategy):
    ret = []
    for symbols, intervals in zip(zip(*datasources), strategy.datas_struct):
        symbols_by_db = defaultdict(set)
        for db, _symbol in symbols:
            symbols_by_db[db].add(_symbol)
        for db, _symbols_by_db in symbols_by_db.items():
            ret.append((db, 'prepare', list(_symbols_by_db), intervals))
    return ret


def _batch_call_backend_method(arguments):
    async_funcs = []
    for backend, funcname, *args in arguments:
        func = getattr(backend, funcname)
        if iscoroutinefunction(func):
            async_funcs.append(asyncio.create_task(func(*args)))
        else:
            func(*args) # 准备好这些数据字段，一直等待准备好为止
    if async_funcs:
        for f in asyncio.as_completed(async_funcs):
            pass


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
