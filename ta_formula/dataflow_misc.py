from collections import defaultdict

from .datasource import get_backend
from .exceptions import DatasListNotMatch


class RequestParser:
    def __init__(self, request):
        self.request = request
        self.datasources = []

    def get_strategy_params(self):
        pass

    def get_prepare_data_params(self, strategy):
        pass


class DictRequestParser(RequestParser):
    def get_strategy_params(self):
        return (
            self.request["pyx_file"],
            self.request["params"],
            self.request["return_fields"],
        )

    def get_prepare_data_params(self, strategy):
        # datasources: `[["{symbol}@{bid}", ...], ...]`
        datasources = self.request["datasources"]
        datas_struct = self.request.get("datas", strategy.datas_struct)
        for one_unit_sources in datasources:
            if len(datas_struct) != len(one_unit_sources):
                raise DatasListNotMatch(
                    "request params datas and datasources not matched"
                )
            b = []
            for source, intervals in zip(one_unit_sources, datas_struct):
                symbol, bid = source.split("@")
                b.append((get_backend(bid), symbol, intervals))
            self.datasources.append(b)
        # self.datasources: `[[(backend, symbol, intervals), ...], ...]`
        ret = []
        for group_unit_sources in zip(*self.datasources):
            intervals = group_unit_sources[0][2]
            symbols_by_db = defaultdict(set)
            for source in group_unit_sources:
                symbols_by_db[source[0]].add(source[1])
            for db, symbols in symbols_by_db.items():
                ret.append((db, list(symbols), intervals))
        return ret
