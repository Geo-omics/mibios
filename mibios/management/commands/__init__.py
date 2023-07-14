class WriteMixin:
    """
    Mixin for Command classes with output writing convenience methods

    Offers print-like color-styled methods (success, info, notice, warn, err)
    to write via the command's stdout and stderr.
    """
    _sep = ' '

    def _write(
        self,
        *txt,
        sep=_sep,
        end=None,
        file=None,
        style=None,
        flush=None,
    ):
        if file is None:
            file = self.stdout

        kwargs = {}
        if end is not None:
            kwargs['ending'] = end

        txt = sep.join((i if isinstance(i, str) else str(i) for i in txt))
        if style is not None:
            txt = style(txt)

        file.write(txt, **kwargs)

        # flush if non-newline end is given, but kw overrides
        if flush is None and end not in (None, '\n'):
            flush = True
        if flush:
            file.flush()

    def success(self, *txt, **kwargs):
        if 'style' not in kwargs:
            kwargs['style'] = self.style.SUCCESS
        self._write(*txt, **kwargs)

    def info(self, *txt, **kwargs):
        # no style for info
        self._write(*txt, **kwargs)

    def warn(self, *txt, **kwargs):
        if 'style' not in kwargs:
            kwargs['style'] = self.style.WARNING
        self._write(*txt, **kwargs)

    def err(self, *txt, **kwargs):
        if 'style' not in kwargs:
            kwargs['style'] = self.style.ERROR
        if 'file' not in kwargs:
            kwargs['file'] = self.stderr
        self._write(*txt, **kwargs)

    def notice(self, *txt, **kwargs):
        if 'style' not in kwargs:
            kwargs['style'] = self.style.NOTICE
        self._write(*txt, **kwargs)
