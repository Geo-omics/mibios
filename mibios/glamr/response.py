import gc

from django.http.response import HttpResponse as DjangoHttpResponse
from django.template.response import TemplateResponse as DjangoTemplateResponse


class ExtraCloserMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resource_closers.append(self._extra_close)

    def _extra_close(self):
        num = gc.collect()


class HttpResponse(ExtraCloserMixin, DjangoHttpResponse):
    pass


class TemplateResponse(ExtraCloserMixin, DjangoTemplateResponse):
    pass
