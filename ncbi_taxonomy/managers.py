from pathlib import Path

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db.models import ForeignObjectRel

from . import DUMP_FILES
from mibios.umrad.utils import atomic_dry, InputFileSpec
from mibios.umrad.manager import InputFileError, Loader as UMRADLoader


class DumpFile(InputFileSpec):
    def setup(self, loader, column_specs=None, path=None):
        """
        Automatically setup the spec based on model
        """
        self.has_header = False  # let's just not rely on the auto-detection
        if column_specs is None:
            # We expect the models to have exactly one field per dump file
            # column.  Assume, that at spec object initialization only
            # pre-processing functions were passed as arguments in two-item
            # format, e.g. ('field_name', procfunc)
            if self._spec is None:
                preprocs = {}
            else:
                preprocs = dict(self._spec)
            column_specs = []

            # get all fields defined by our model in order
            fields = loader.model.get_fields(skip_auto=True, with_m2m=True).fields  # noqa: E501
            for i in fields:
                if isinstance(i, ForeignObjectRel):
                    # filter out reverse fields
                    continue
                if i.name in preprocs:
                    column_specs.append((i.name, preprocs[i.name]))
                else:
                    column_specs.append(i.name)

        super().setup(loader, column_specs=column_specs, path=path)

    def iterrows(self):
        """
        An iterator over NCBI taxonomy dump rows

        Files don't have headers, columns are separated by <tab>|<tab>, lines
        may end in <tab>|
        """
        with self.path.open() as f:
            print(f'File opened: {f.name}')
            for line in f:
                line = line.rstrip('\n')
                if line.endswith('\t|'):
                    # most lines, strip "\t|"
                    line = line[:-2]
                row = line.split('\t|\t')
                row = [i.strip() for i in row]  # TODO: is this needed?
                yield row


class Loader(UMRADLoader):
    default_load_kwargs = dict(bulk=True, update=True)

    def setup_spec(self, spec=None, **kwargs):
        if spec is None and self.spec is None:
            # as all ncbi tax model share the loader class, here each loader
            # will get it's own instance of the spec
            spec = DumpFile()
        super().setup_spec(spec=spec, **kwargs)

    def get_file(self):
        try:
            path = settings.NCBI_TAXONOMY_DUMP_DIR
        except AttributeError as e:
            raise ImproperlyConfigured(f'{e.name} setting not configured')
        return Path(path) / DUMP_FILES[self.model._meta.model_name]


class CitationLoader(Loader):

    def parse_nodes(self, value, obj):
        if value is None:
            return None

        try:
            value = [(int(i),) for i in value.split()]
        except ValueError as e:
            raise InputFileError(f'failed parsing node column: {e}')
        return value

    spec = DumpFile(
        ('node', parse_nodes),
    )


class TaxNodeLoader(Loader):
    @atomic_dry
    def load(self, **kwargs):
        super().load(**kwargs)

        # super().load() does not support FKs on self, so it's done below, w/o
        # support for validation/diffs/limit etc.
        print('re-loading taxnodes...', end='', flush=True)
        objs = self.model.objects.all()
        by_taxid = {i.taxid: i for i in objs}
        print('[OK]')
        print('importing parent-child relations...')
        for row in self.spec.iterrows():
            taxid, parentid, *_ = row
            by_taxid[int(taxid)].parent = by_taxid[int(parentid)]
        self.fast_bulk_update(objs, fields=['parent'])
