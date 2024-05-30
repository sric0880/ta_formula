import asyncio
import logging
import threading
from collections import defaultdict
from functools import singledispatch
from inspect import iscoroutinefunction

__all__ = [
    'DataBackend',
    'AioDataBackend',
    'add_backend',
    'get_backend',
    'get_backends',
    'close_all_backends',
    'close_all_backends_async',
]

_registered_backends = {}
data_backends = {}

class _BaseDataBackend:
    def __init_subclass__(cls) -> None:
        _registered_backends[cls.__name__] = cls

    def __init__(self, bid, config) -> None:
        self.bid = bid
        self.config = config

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.bid}, {self.config})'

class DataBackend(_BaseDataBackend):
    def __init__(self, bid, config) -> None:
        super().__init__(bid, config)
        self._conditions = defaultdict(threading.Semaphore)

    def prepare(self, out: dict, symbols: list, intervals: list):
        '''
        需要准备`symbols`下的所有`intervals`数据

        每次策略开始前只调用一次，如果已经准备好的数据，再次准备无效

        `symbols`是金融产品的代码列表

        `intervals`是数据频率，可以是'1m','1D', 或者'tick'

        所有准备好的数据保存在out中
        '''
        pass

    def close(self):
        pass

class AioDataBackend(_BaseDataBackend):
    def __init__(self, bid, config) -> None:
        super().__init__(bid, config)

    async def prepare(self, out: dict, symbols: list, intervals: list):
        '''
        需要准备`symbols`下的所有`intervals`数据

        每次策略开始前只调用一次，如果已经准备好的数据，再次准备无效

        `symbols`是金融产品的代码列表

        `intervals`是数据频率，可以是'1m','1D', 或者'tick'

        所有准备好的数据保存在out中
        '''

    async def close(self):
        pass

# singledispatch 3.9暂时不支持Union类型
# 从3.11开始支持：https://docs.python.org/3/library/functools.html
# DataBackendClass = Union[DataBackend, AioDataBackend]

@singledispatch
def add_backend(claname: str, bid: str, config: dict):
    if bid not in data_backends:
        cls = _registered_backends[claname]
        data_backends[bid] = backend = cls(bid, config)
        return backend
    else:
        return data_backends[bid]

@add_backend.register
def _(instance: DataBackend):
    if instance.bid not in data_backends:
        data_backends[instance.bid] = instance
    return instance

@add_backend.register
def _(instance: AioDataBackend):
    data_backends[instance.bid] = instance
    return instance

def get_backend(bid: str):
    try:
        return data_backends[bid]
    except KeyError as e:
        raise RuntimeError(f'{bid} data backend is not registered') from e


def get_backends(bids):
    return [get_backend(bid) for bid in bids]


def close_all_backends():
    async_closes = []
    for bid, backend in data_backends.items():
        if iscoroutinefunction(backend.close):
            async_closes.append(asyncio.create_task(backend.close(), bid=bid))
        else:
            backend.close()
            logging.info(f"数据后台 {bid} 关闭")
    if async_closes:
        for f in asyncio.as_completed(async_closes):
            logging.info(f"数据后台 {f.get_name()} 关闭")
    data_backends.clear()


async def close_all_backends_async():
    async_closes = []
    for bid, backend in data_backends.items():
        if iscoroutinefunction(backend.close):
            async_closes.append(asyncio.create_task(backend.close(), bid=bid))
        else:
            async_closes.append(
                asyncio.get_event_loop().run_in_executor(None, backend.close)
            )
        logging.info(f"数据后台 {bid} 关闭")
    if async_closes:
        await asyncio.gather(async_closes)
    data_backends.clear()
