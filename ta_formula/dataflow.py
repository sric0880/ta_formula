import asyncio
import logging
from collections import defaultdict
from concurrent.futures import Future, wait
from contextlib import asynccontextmanager, contextmanager
from inspect import iscoroutinefunction

from .datasource import get_backend

__all__ = ['open_dataflow_async', 'open_dataflow']


@asynccontextmanager
async def open_dataflow_async(datasources, strategy):
    datasources = _parse_datasources(datasources)
    call_prepare_args = _prepare_arguments(datasources, strategy)
    out = await _batch_call_backend_method_async(call_prepare_args) # 准备所有需要的数据
    try:
        if is_streaming: # 订阅，数据随时间依次到达
            def datas_generator():
                fut = Future()
                while True:
                    yield
            yield datas_generator
        else: # 数据在prepare时一次性加载完
            datas_lists = []
            for dscfgs in datasources:
                datas = []
                for (db, symbol), interval_list in zip(_parse_datasources(dscfgs), intervals):
                    datas.append([db.get_data(symbol, interval) for interval in interval_list])
                datas_lists.append(datas)
            yield datas_lists # 返回所有需要的数据
    except:
        logging.error(f"strategy calculate error", exc_info=True)
    finally:
        pass
        # await db_func('release') # 释放数据


@contextmanager
def open_dataflow(datasources, strategy):
    datasources = _parse_datasources(datasources)
    call_prepare_args = _prepare_arguments(datasources, strategy)
    out = _batch_call_backend_method(call_prepare_args) # 准备所有需要的数据
    try:
        def _datas_generator():
            for dss in datasources:
                datas = []
                for (db, symbol), intervals in zip(dss, strategy.datas_struct):
                    datas.append([out[db.bid][symbol][interval] for interval in intervals])
                yield datas
        yield _datas_generator()
    except:
        logging.error(f"strategy calculate error", exc_info=True)
    finally:
        # db_func('release') # 释放数据
        pass


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
    out = {}
    async_funcs = []
    for backend, funcname, *args in arguments:
        dbout = out.setdefault(backend.bid, {})
        func = getattr(backend, funcname)
        if iscoroutinefunction(func):
            async_funcs.append(asyncio.create_task(func(dbout, *args)))
        else:
            func(dbout, *args) # 准备好这些数据字段，一直等待准备好为止
    if async_funcs:
        for f in asyncio.as_completed(async_funcs):
            pass
    return out


async def _batch_call_backend_method_async(arguments):
    out = {}
    async_funcs = []
    for backend, funcname, *args in arguments:
        dbout = out.setdefault(backend.bid, {})
        func = getattr(backend, funcname)
        if iscoroutinefunction(func):
            async_funcs.append(asyncio.create_task(func(dbout, *args)))
        else:
            async_funcs.append(
                asyncio.get_event_loop().run_in_executor(None, func, dbout, *args)
            )
    if async_funcs:
        await asyncio.gather(async_funcs)
    return out


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
