import asyncio
import logging
from collections import defaultdict
from functools import singledispatch
from inspect import iscoroutinefunction

from .exceptions import DataBackendNotFound

__all__ = [
    "DataBackend",
    "AioDataBackend",
    "add_backend",
    "get_backend",
    "get_backends",
    "close_all_backends",
    "close_all_backends_async",
]

_registered_backends = {}
data_backends = {}


class _BaseDataBackend:
    def __init_subclass__(cls) -> None:
        _registered_backends[cls.__name__] = cls

    def __init__(self, bid, config) -> None:
        self.bid = bid
        self.config = config
        self._all_datas = defaultdict(dict)
        self._calc_units = set()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.bid}, {self.config})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.bid})"

    def on_update(self, symbol, interval, update_dt=0):
        for unit in list(self._calc_units):
            unit.on_update(self.bid, symbol, interval, update_dt)

    def add_data(self, symbol, interval, data):
        if not isinstance(data, dict):
            return
        # 不能覆盖以前的数据
        if interval in self._all_datas[symbol]:
            return
        self._all_datas[symbol][interval] = data


class DataBackend(_BaseDataBackend):
    def __init__(self, bid, config) -> None:
        super().__init__(bid, config)

    def prepare(self, symbols: list, intervals: list):
        """
        需要准备`symbols`下的所有`intervals`数据，准备好的数据调用`add_data`更新

        每次策略开始前只调用一次，数据准备准备好之后，需要调用`on_update`通知策略

        如果是多线程或异步更新数据，可以准备`{}`空数据，在数据到达之后，再更新这个`{}`并调用`on_update`

        参数：
            `symbols`: 是金融产品的代码列表
            `intervals`: 是数据频率，可以是'1m','1D', 或者'tick'
        """

    def close(self):
        pass


class AioDataBackend(_BaseDataBackend):
    def __init__(self, bid, config) -> None:
        super().__init__(bid, config)

    async def prepare(self, symbols: list, intervals: list):
        """
        需要准备`symbols`下的所有`intervals`数据，准备好的数据调用`add_data`更新

        每次策略开始前只调用一次，数据准备准备好之后，需要调用`on_update`通知策略

        如果是多线程或异步更新数据，可以准备`{}`空数据，在数据到达之后，再更新这个`{}`并调用`on_update`

        参数：
            `symbols`: 是金融产品的代码列表
            `intervals`: 是数据频率，可以是'1m','1D', 或者'tick'
        """

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
        logging.debug(f"{backend} Added")
        return backend
    else:
        return data_backends[bid]


@add_backend.register
def _(instance: DataBackend):
    if instance.bid not in data_backends:
        data_backends[instance.bid] = instance
        logging.debug(f"{instance} Added")
    return data_backends[instance.bid]


@add_backend.register
def _(instance: AioDataBackend):
    if instance.bid not in data_backends:
        data_backends[instance.bid] = instance
        logging.debug(f"{instance} Added")
    return data_backends[instance.bid]


def get_backend(bid: str):
    try:
        return data_backends[bid]
    except KeyError as e:
        raise DataBackendNotFound(f"{bid} data backend is not registered") from e


def get_backends(bids):
    return [get_backend(bid) for bid in bids]


def close_all_backends():
    for bid, backend in data_backends.items():
        backend.close()
        logging.debug(f"{backend!r} Shutdown")
    data_backends.clear()


async def close_all_backends_async():
    async_closes = []
    for bid, backend in data_backends.items():
        if iscoroutinefunction(backend.close):
            async_closes.append(asyncio.create_task(backend.close()))
        else:
            async_closes.append(
                asyncio.get_event_loop().run_in_executor(None, backend.close)
            )
        logging.debug(f"{backend!r} Shutdown")
    if async_closes:
        await asyncio.gather(*async_closes)
    data_backends.clear()
