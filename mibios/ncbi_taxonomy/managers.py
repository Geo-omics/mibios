from collections import defaultdict
from pathlib import Path

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db.models import ForeignObjectRel

from . import DUMP_FILES
from mibios.umrad.utils import atomic_dry, InputFileSpec
from mibios.umrad.manager import InputFileError, Loader as UMRADLoader
from mibios.umrad.model_utils import delete_all_objects_quickly
from mibios.umrad.utils import ProgressPrinter


class DumpFile(InputFileSpec):
    def setup(self, loader, column_specs=None, **kwargs):
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
                if i.name in loader.spec_skip_fields:
                    continue
                if isinstance(i, ForeignObjectRel):
                    # filter out reverse fields
                    continue
                if i.name in preprocs:
                    column_specs.append((i.name, preprocs[i.name]))
                else:
                    column_specs.append(i.name)

        super().setup(loader, column_specs=column_specs, **kwargs)

    def iterrows(self):
        """
        An iterator over NCBI taxonomy dump rows

        Files don't have headers, columns are separated by <tab>|<tab>, lines
        may end in <tab>|
        """
        with self.file.open() as f:
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

    spec_skip_fields = []
    """ Fields that don't get loaded via dump file, the spec setup will ignore
    them """

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


class TaxNameLoader(Loader):
    @atomic_dry
    def load(self, update=True, **kwargs):
        """
        Load ncbi tax names

        Update mode: Using the parent Loader update mode won't work well for
        names as the first column is not a unique key or similar.  Since
        TaxName is a leaf model (it should not be target of an FK) we'll just
        delete it all and load from scratch.
        """
        if update:
            print('NOTICE: update mode for tax names works by deleting and '
                  'replacing all existing data')
            delete_all_objects_quickly(self.model)
        super().load(update=False, **kwargs)
        print('Populating nodes with scientific names...')
        self.populate_sci_names()
        print('[OK]')

    @atomic_dry
    def populate_sci_names(self):
        """
        Populate the TaxNode.name field with scientific names
        """
        TaxName = self.model
        TaxNode = TaxName._meta.get_field('node').related_model
        names = dict(
            TaxName.objects.filter(name_class=TaxName.NAME_CLASS_SCI)
            .values_list('node_id', 'name')
        )
        print(f'Loaded {len(names)} scientific names')
        nodes = TaxNode.objects.only('pk', 'name')
        print(f'Loaded {len(nodes)} tax nodes')
        pp = ProgressPrinter('{progress} tax nodes processed')
        new = changed = 0
        for obj in pp(nodes):
            new_name = names[obj.pk]
            if obj.name == '':
                new += 1
            elif new_name != obj.name:
                changed += 1
            obj.name = new_name
        print(f'Sci. names newly set: {new} / exiting but changed: {changed}')
        TaxNode.loader.fast_bulk_update(nodes, ['name'])


class TaxNodeLoader(Loader):
    spec_skip_fields = ['ancestors', 'name']

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
        print('Populating ancestry...')
        self.populate_ancestry()
        print('[OK]')

    def lineages(self, tree, *ancestors):
        """
        traverse tree left-to-right and yield full lineages

        This is a helper for populating the ancestry relation
        """
        if tree:
            for pk, subtree in tree.items():
                yield from self.lineages(subtree, *ancestors, pk)
        else:
            yield ancestors

    def ancestry_node_pairs(self, lineages):
        """
        Generate distinct (ancestor, descendend) pk pairs

        This is a helper to populate the ancestry relation.  The lineages
        parameter must be an iterable of all the lineages ordered left-to-right
        (or right-to-left.)
        """
        # whenever some initial nodes are the same as for the previous lineage,
        # the links between these can be skipped as we got them last time.  The
        # index of the first fresh node is saved in the first_change_index
        # variable
        prev = []
        for lin in lineages:
            first_change_index = 0
            for first_change_index, (a, b) in enumerate(zip(prev, lin)):
                if a != b:
                    break

            for i in range(first_change_index, len(lin)):
                for j in range(i):
                    yield lin[j], lin[i]

            prev = lin

    @atomic_dry
    def populate_ancestry(self):
        """
        Populate the ancestor m2m relation

        Assumes that there is a root node with taxid 1.
        """
        m2m_field = self.model._meta.get_field('ancestors')
        Through = m2m_field.remote_field.through
        delete_all_objects_quickly(Through)

        root = self.get(taxid=1)

        print('  Generating tree from parent relations...', end='', flush=True)
        # some super clever code to turn the parent relation into a tree (a
        # dict of dicts, keys are PKs, values are sub-trees
        dict_of_dict = lambda: defaultdict(dict)  # noqa: E731
        forest = defaultdict(dict_of_dict)
        for child, parent in self.values_list('pk', 'parent_id').iterator():
            forest[parent][child] = forest[child]

        # Pick the tree for root.  After this, the root, is implicit, while we
        # keep the root node itself, we will only work with the tree below
        # root, so the ancestor relation with it will not be saved.
        tree = forest[root.pk]
        del forest
        # in NCBI taxonomy root's parent is root, that circle must be removed,
        # as we'll do some depth-first tranversal, so we want a real tree
        del tree[root.pk]
        print(' [OK]')

        lins = self.lineages(tree)
        pairs = self.ancestry_node_pairs(lins)
        through_objs = ((
            Through(from_taxnode_id=b, to_taxnode_id=a)
            for a, b in pairs
        ))
        self.bulk_create_wrapper(Through.objects.bulk_create)(through_objs)
