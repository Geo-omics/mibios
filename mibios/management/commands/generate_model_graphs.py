from pathlib import Path

from django.core.management.base import BaseCommand

from mibios.model_graphs import make_all_graphs


class Command(BaseCommand):
    help = 'Generate model graphs'

    def add_arguments(self, argp):
        argp.add_argument(
            '--output-dir',
            default=None,
            help='Name of output directory.  By default the images will be '
                 'saved under the static root so they can be served via the '
                 'web.'
        )

    def handle(self, *args, output_dir=None, **options):
        if output_dir is not None:
            output_dir = Path(output_dir)
        images = make_all_graphs(output_dir)
        if images:
            self.stdout.write('Model graph image files saved as:')
            for i in images:
                self.stdout.write(str(i))
