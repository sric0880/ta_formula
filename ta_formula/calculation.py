import logging
import time
from collections import defaultdict

# from line_profiler import LineProfiler

# profile = LineProfiler()

__all__ = ["calculation_center"]
# __all__ = ['calculation_center', 'profile']


class _CalculationCenter:
    def __init__(self, lock=None) -> None:
        self.calc_units = {}
        # 引用计数需要原子操作
        self._lock = lock
        self.ref_counts = defaultdict(int)

    def push(self, strategy, symbol_infos):
        _hash = hash(
            (
                strategy._hash,
                tuple(
                    (db.bid, symbol, *intervals)
                    for db, symbol, intervals in symbol_infos
                ),
            )
        )
        if self._lock is not None:
            self._lock.acquire()
        self.ref_counts[_hash] += 1
        unit = self.calc_units.get(_hash, None)
        if unit is None:
            self.calc_units[_hash] = unit = _CalculateUnit(
                strategy, symbol_infos, _hash
            )
            logging.debug(f"{unit}: New. Ref count {self.ref_counts[_hash]}")
        else:
            logging.debug(f"{unit}: Ref plus. Count {self.ref_counts[_hash]}")
        if self._lock is not None:
            self._lock.release()
        return unit

    def pop(self, unit):
        _hash = unit._hash
        if self._lock is not None:
            self._lock.acquire()
        ref = self.ref_counts[_hash]
        if ref == 0:
            logging.error(f"{unit}: Ref is zero, cannot pop.")
            return
        ref -= 1
        self.ref_counts[_hash] = ref
        if ref == 0:
            logging.debug(f"{unit}: Delete. Ref count {ref}")
            unit = self.calc_units.pop(_hash)
            unit.release()
        else:
            logging.debug(f"{unit}: Ref minus. Count {ref}")
        if self._lock is not None:
            self._lock.release()


class _ConsistentFlag:
    __slots__ = ["states", "_full_state"]

    def __init__(self, full_state) -> None:
        self._full_state = full_state
        self.states = [False] * full_state

    def consistent(self):
        first_state = self.states[0]
        for state in self.states:
            if state != first_state:
                return False
        return True

    def reset(self):
        self.states = [False] * self._full_state


class _Data:
    __slots__ = ["cflag", "data", "index"]

    def __init__(self, cflag, data, index) -> None:
        self.cflag = cflag
        self.data = data
        self.index = index


class _CalculateUnit:
    def __init__(self, strategy, symbol_infos, _hash) -> None:
        self.strategy = strategy
        self.symbol_infos = symbol_infos
        self._hash = _hash
        self._datas_list = [None] * len(strategy.datas_interface)
        self.datas_map = {}
        self._symbol_names = []
        self._cflags = []
        self._waiters = []
        datas = []
        # 可能不同交易所有相同的symbol
        for db, symbol, intervals in symbol_infos:
            self._symbol_names.append((db.bid, symbol))
            _datas = [db._all_datas[symbol][interval] for interval in intervals]
            datas.append(_datas)
            cflag = _ConsistentFlag(len(intervals))
            self._cflags.append(cflag)
            symbols = self.datas_map.setdefault(db.bid, {})
            symbols[symbol] = dict(
                zip(
                    intervals,
                    [_Data(cflag, _data, i) for i, _data in enumerate(_datas)],
                )
            )
            db._calc_units.add(self)
        strategy.feed_external_datas(datas, self._datas_list)

    def release(self):
        for db, _, _ in self.symbol_infos:
            try:
                db._calc_units.remove(self)
            except KeyError:
                pass

    # @profile
    def on_update(self, backend_id, symbol, interval, update_dt):
        # TODO: 如果有多个数据后台多线程运行，这个函数需要考虑线程安全
        try:
            _data = self.datas_map[backend_id][symbol][interval]
        except KeyError:
            return
        # 附加 数据接收时间戳
        start_counter = time.perf_counter_ns()
        data_rec_time = int(time.time() * 1000000)
        _data.cflag.states[_data.index] = True

        # 同一标的的不同频率的数据，要么全部没更新，要么全部更新
        for cflag in self._cflags:
            if not cflag.consistent():
                return

        # 全部重置
        for cflag in self._cflags:
            cflag.reset()
        # 计算入口
        signal = self.strategy.calculate_x(self._datas_list)
        # 附加 计算单元ID
        signal["calc_unit_id"] = self._hash
        signal["data_update_time"] = update_dt
        signal["data_rec_time"] = data_rec_time
        signal["calc_time"] = start_counter
        signal["symbols"] = self._symbol_names
        for waiter in self._waiters:
            waiter.add_result(self._hash, signal)

    def _add_waiter(self, waiter):
        self._waiters.append(waiter)

    def _remove_waiter(self, waiter):
        self._waiters.remove(waiter)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, value: object) -> bool:
        return id(self) == id(value)

    def __str__(self):
        return f"{self.__class__.__name__}({self.strategy}, {self.symbol_infos}, {self._hash})"
