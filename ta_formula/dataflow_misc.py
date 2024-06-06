from collections import defaultdict

from .datasource import get_backend


def _prepare_arguments(datasources, datas_struct):
    ret = []
    for symbols, intervals in zip(zip(*datasources), datas_struct):
        symbols_by_db = defaultdict(set)
        for db, _symbol in symbols:
            symbols_by_db[db].add(_symbol)
        for db, _symbols_by_db in symbols_by_db.items():
            ret.append((db, 'prepare', list(_symbols_by_db), intervals))
    return ret


def _parse_datasources(datasources: list):
    '''
    parse like '[[{data_backend_clsname}.{bid}.{symbol}, ...], ...]'
    '''
    ret = []
    for dss in datasources:
        _dss = []
        for ds in dss:
            bid, symbol = ds.split('.')
            _dss.append((get_backend(bid), symbol))
        ret.append(_dss)
    return ret
