class TaFormulaError(Exception):
    pass


class DataBackendNotFound(TaFormulaError):
    pass


class PyxSyntaxError(TaFormulaError):
    pass


class DatasListNotMatch(TaFormulaError):
    pass


class TaTimeoutError(TaFormulaError):
    pass
