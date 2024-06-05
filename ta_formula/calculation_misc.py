class _ConsistentFlag:
    __slots__ = ['states', '_full_state']
    def __init__(self, full_state) -> None:
        self._full_state = full_state
        self.states = [False]*full_state

    def consistent(self):
        first_state = self.states[0]
        for state in self.states:
            if state != first_state:
                return False
        return True

    def reset(self):
        self.states = [False]*self._full_state

class _Data:
    __slots__ = ['cflag', 'data', 'index']
    def __init__(self, cflag, data, index) -> None:
        self.cflag = cflag
        self.data = data
        self.index = index
