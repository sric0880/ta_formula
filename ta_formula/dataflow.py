import asyncio
import logging
import threading
from collections import defaultdict
from concurrent import futures
from inspect import iscoroutinefunction

from .datasource import get_backend
from .strategy import get_strategy

__all__ = ['open_signal_stream']

class _CalculationCenter:
    def __init__(self) -> None:
        self.calc_units = {}
        # 引用计数需要原子操作
        self._lock = threading.Lock()
        self.ref_counts = defaultdict(int)

    def push(self, strategy, symbol_infos):
        _hash = hash((strategy._hash, tuple((db.bid, symbol, *intervals) for db, symbol, intervals in symbol_infos)))
        with self._lock:
            self.ref_counts[_hash] += 1
            unit = self.calc_units.get(_hash, None)
            if unit is None:
                self.calc_units[_hash] = unit = _CalculateUnit(strategy, symbol_infos, _hash)
                logging.debug(f'{unit}: New. Ref count {self.ref_counts[_hash]}')
            else:
                logging.debug(f'{unit}: Ref plus. Count {self.ref_counts[_hash]}')
        return unit

    def pop(self, unit):
        _hash = unit._hash
        with self._lock:
            ref = self.ref_counts[_hash]
            if ref == 0:
                logging.error(f'{unit}: Ref is zero, cannot pop.')
                return
            ref -= 1
            self.ref_counts[_hash] = ref
            if ref == 0:
                logging.debug(f'{unit}: Delete. Ref count {ref}')
                unit = self.calc_units.pop(_hash)
                unit.release()
            else:
                logging.debug(f'{unit}: Ref minus. Count {ref}')

calculation_center = _CalculationCenter()


class _CalculateUnit:
    def __init__(self, strategy, symbol_infos, _hash) -> None:
        self.fut = futures.Future()
        self.strategy = strategy
        self.symbol_infos = symbol_infos
        self._hash = _hash
        self.datas_list = [None]*len(strategy.datas_interface)
        self.datas_consistent = {}
        datas = []
        for db, symbol, intervals in symbol_infos:
            self.datas_consistent[symbol] = (len(intervals), { interval: 0 for interval in intervals})
            db._calc_units.add(self)
            datas.append([db._all_datas[symbol][interval] for interval in intervals])
        strategy.feed_external_datas(datas, self.datas_list)

    def release(self):
        for db, _, _ in self.symbol_infos:
            try:
                db._calc_units.remove(self)
            except KeyError:
                pass

    def on_update(self, symbol, interval):
        intervals = self.datas_consistent.get(symbol, None)
        if not intervals:
            return
        intervals = intervals[1]
        if interval not in intervals:
            return
        intervals[interval] = 1

        if self._is_datas_consistent():
            self._reset_datas_consistent()
            signal = self.strategy.calculate_x(self.datas_list)
            self.fut.set_result(signal)
            new_fut = futures.Future()
            new_fut._waiters = self.fut._waiters
            self.fut = new_fut

    def _is_datas_consistent(self):
        ''' 同一标的的不同频率的数据，要么全部没更新，要么全部更新'''
        all_zero = True
        for full_count, intervals in self.datas_consistent.values():
            state = sum(intervals.values())
            if state > 0:
                if state < full_count:
                    return False
                all_zero = False
        if all_zero:
            return False
        return True

    def _reset_datas_consistent(self):
        for _, intervals in self.datas_consistent.values():
            for interval in intervals.keys():
                intervals[interval] = 0

    def _add_waiter(self, waiter):
        self.fut._waiters.append(waiter)

    def _remove_waiter(self, waiter):
        with self.fut._condition:
            self.fut._waiters.remove(waiter)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, value: object) -> bool:
        return id(self) == id(value)

    def __str__(self):
        return f'{self.__class__.__name__}({self.strategy}, {self.symbol_infos}, {self._hash})'

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
                    yield fut.result()
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
