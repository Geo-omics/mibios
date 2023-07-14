from time import monotonic

from django.core.management.base import CommandError
from django.db import connections
from django.db.transaction import atomic

from mibios.management.commands import WriteMixin
from mibios.management.commands.dump_data import DumpBaseCommand, MODE_ADD
from mibios.omics import get_sample_model
from mibios.omics.models import Alignment, Contig, Gene


class Command(WriteMixin, DumpBaseCommand):
    help = 'dump omics data for given sample(s)'

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            'sample_ids',
            metavar='SAMPLE',
            nargs='+',
            help='One or more sample_id values',
        )
        parser.set_defaults(mode=MODE_ADD)

    @atomic
    def do_dump(self, sample_ids=None, **options):
        Sample = get_sample_model()
        samples = Sample.objects.filter(sample_id__in=sample_ids)

        if len(samples) != len(sample_ids):
            raise CommandError('some samples not found')

        Contig_taxa = Contig._meta.get_field('taxa').remote_field.through
        Gene_taxa = Gene._meta.get_field('taxa').remote_field.through

        qsets = [
            Contig.objects.filter(sample__in=samples),
            Contig_taxa.objects.filter(contig__sample__in=samples),
            Gene.objects.filter(sample__in=samples),
            Gene_taxa.objects.filter(gene__sample__in=samples),
            Alignment.objects.filter(gene__sample__in=samples),
        ]

        with connections['default'].cursor() as cur:
            for i in qsets:
                self.info(f'{i.model._meta.model_name}... ', end='')
                t0 = monotonic()
                self.dump_from_queryset(cur, i)
                t1 = monotonic()
                rate = cur.rowcount / (t1 - t0)
                self.info(f'{cur.rowcount} row ({rate:.0f}/s) ', end='')
                self.success('[OK]')
