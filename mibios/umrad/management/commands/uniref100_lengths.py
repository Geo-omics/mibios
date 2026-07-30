from django.core.management.base import BaseCommand, CommandError

from mibios.umrad.models import UniRef100


class Command(BaseCommand):
    help = 'Compile UniRef100 access to length mapping'

    def add_arguments(self, parser):
        super().add_arguments(parser)
        subs = parser.add_subparsers(dest='cmd', required=True)
        build_p = subs.add_parser(
            'build',
            help='build reference length file.  Output is written to stdout.',
        )
        build_p.add_argument(
            'sequence_file',
            help='UniRef100 sequence file.  This is a file with ascii-encoded '
                 'upper-case amino acid sequences separated by the two bytes '
                 '0x0a00.',
        )
        build_p.add_argument(
            'accession-file',
            help='UniRef100 accession mapping file.  This is a 3-column '
                 'tab-separated text file, mapping sequence number to accession. '
                 'Starts with 0.  Third column is always 0.'
        )
        check_p = subs.add_parser(
            'check',
            help='Check length stored in DB against reference',
        )
        check_p.add_argument('length_file', help='Length file made with build command')
        check_p.add_argument('--batch-size', type=int, help='Batch size')
        load_p = subs.add_parser(
            'load',
            help='Load UniRef100 lengths from lengths file made with build command.',
        )
        load_p.add_argument('length_file', help='Length file made with build command')
        load_p.add_argument('-n', '--dry-run', action='store_true',
                            help='Do not make any changes to the DB.')
        load_p.add_argument(
            '--missing-only', action='store_true',
            help='Only add missing lengths.  Existing length data remains',
        )

    def handle(self, **options):
        match options['cmd']:
            case 'build':
                accessions = self.get_accessions(options['accession_file'])
                lengths = self.get_lengths(options['sequence_file'])
                for accession, length in zip(accessions, lengths):
                    self.stdout.write(f'{accession}\t{length}\n')
            case 'check':
                if batch_size := options['batch_size']:
                    if batch_size < 1_000_000:
                        raise CommandError('that\'s too small for a batch size')
                    else:
                        batch_size = None  # use default

                UniRef100.loader.check_lengths(
                    options['length_file'],
                    verbose=options['verbosity'] >= 2,
                    batch_size=batch_size,
                )
            case 'load':
                UniRef100.loader.load_lengths(
                    options['length_file'],
                    dry_run=options['dry_run'],
                    missing_only=options['missing_only'],
                )

    def get_accessions(self, path):
        with open(path, 'r') as ifile:
            for lnum, line in enumerate(ifile):
                try:
                    seqnum, accn, third = line.rstrip('\n').split('\t')
                except ValueError as e:
                    raise CommandError(f'failed parsing 0-line {lnum}: {e}')

                try:
                    seqnum = int(seqnum)
                except ValueError as e:
                    raise CommandError(f'failed parsing seq num at 0-line {lnum}: {e}')

                if seqnum != lnum:
                    raise CommandError('unexpected seqnum {seqnum} at 0-line {lnum}')

                if third != '0':
                    raise CommandError('third column has {third} at 0-line {lnum}')

                yield accn

    def get_lengths(self, path):
        with open(path, 'rb') as ifile:
            for num, seq in enumerate(ifile):
                seq = seq.rstrip(b'\n')
                if num == 0:
                    pass
                elif seq.startswith(b'\x00'):
                    seq = seq.lstrip(b'\x00')
                else:
                    self.stderr.write('seq at {num} does not start with zero byte\n')

                try:
                    if not seq.decode().isupper():
                        self.stderr.write('seq at {num}: not all upper case\n')
                except UnicodeDecodeError as e:
                    self.stderr.write(f'seq at {num}: not unicode: {e}\n')

                # print(f'BORK {num} {seq}\n')
                yield len(seq)
