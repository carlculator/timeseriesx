
class BaseMixin(object):
    _series = None

    def __init__(self, *args, **kwargs):
        super().__init__()

    def as_pd_series(self):
        pass
