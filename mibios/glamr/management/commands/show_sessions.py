from collections import Counter
from datetime import datetime
from statistics import quantiles

from django.contrib.sessions.models import Session
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = 'Show session information'

    def add_arguments(self, parser):
        parser.add_argument(
            '--from',
            default=None,
            help='Show sessions initiated after given (isoformatted) timestamp'
        )
        parser.add_argument(
            '--to',
            default=None,
            help='Show sessions initiated before given (isoformatted) timestamp'
        )
        parser.add_argument(
            '--details', action='store_true',
            help='Print details of sessions instead of statistics',
        )
        parser.add_argument(
            '--no-filter', action='store_true',
            help='include invalid session',
        )

    def handle(self, *args, outdir=None, **options):
        if options['from']:
            try:
                start = datetime.fromisoformat(options['from']).astimezone()
            except ValueError as e:
                raise CommandError(
                    f'Must pass an isoformatted timestamp via --from: {e}'
                )
        else:
            start = None

        if options['to']:
            try:
                end = datetime.fromisoformat(options['to']).astimezone()
            except ValueError as e:
                raise CommandError(
                    f'Must pass an isoformatted timestamp via --to: {e}'
                )
        else:
            end = None

        num_no_first = 0
        sessions = []
        for obj in Session.objects.all():
            data = obj.get_decoded()
            data['key'] = obj.session_key
            data['expire_date'] = obj.expire_date
            if first := data.get('first_time'):
                first = datetime.fromisoformat(first)
                if start and first < start:
                    continue
                if end and first < end:
                    continue

            if first or options['no_filter']:
                sessions.append(data)
            elif first is None:
                num_no_first += 1

        if num_no_first:
            self.print(f'sessions without first timestamp: {num_no_first}')
        self.print(f'Valid sessions: {len(sessions)}')

        if options['details']:
            self.details(sessions)
            return

        now = datetime.now().astimezone()

        num_expired = 0
        num_bounced_off = 0
        num_via_bouncer = 0
        num_via_entry = 0
        num_requests = []
        paths = Counter()
        second_times = []
        for data in sessions:
            if data['expire_date'] <= now:
                num_expired += 1
            if 'numrequests' in data:
                num_requests.append(data['numrequests'])
            if 'time_to_second' in data:
                second_times.append(data['time_to_second'])
            if data.get('bounced'):
                if data.get('num_requests') == 1:
                    num_bounced_off += 1
                else:
                    num_via_bouncer += 1
            if path := data.get('entrypath'):
                num_via_entry += 1
                paths[path] += 1

        self.print(f'expired: {num_expired}')
        self.print(f'bounced off: {num_bounced_off}')
        if num_requests:
            self.print(f'requests per session: {quantiles(num_requests)} '
                       f'(total:{len(num_requests)})')
        if second_times:
            self.print(f'seconds until second request: {quantiles(second_times)} '
                       f'(total:{len(second_times)})')
        if paths:
            self.print('Most common entry paths:')
            for path, count in paths.most_common()[:10]:
                print(f'{path:>20}: {count:>6}')

    def details(self, sessions):
        now = datetime.now().astimezone()
        for num, data in enumerate(sessions, start=1):
            exptxt = '  (exipred)' if data['expire_date'] <= now else ''
            if num:
                print('-----------------------------')
            print(f'{num:>4}.{"key":>15}: +++ {data.pop("key")} +++')
            print(f'{"exp":>20}: {data.pop("expire_date")}{exptxt}')
            for k, v in data.items():
                print(f'{k:>20}: {v}')

    def print(self, *msg, sep=' ', end='\n'):
        self.stdout.write(sep.join(msg) + end)
