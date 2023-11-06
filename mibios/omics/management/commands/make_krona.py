from django.core.management.base import BaseCommand

from mibios.omics.models import TaxonAbundance


class Command(BaseCommand):
    help = 'Generate all missing Krona charts'

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            '--remake-all',
            action='store_true',
            help='Also remake any existing chart',
        )

    def handle(self, **options):
        TaxonAbundance.objects.make_all_krona_charts(
            keep_existing=not options['remake_all'],
        )
