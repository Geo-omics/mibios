from time import sleep

from mibios.omics.utils import Timestamper
from mibios.umrad.utils import ProgressPrinter


def test_timestamper():
    ts = Timestamper(template='[ {timestamp} testing ]  ',
                     file_copy='/tmp/test_timestamper.txt')
    with ts as obj:
        print('starting up ...')
        print('second line ...', end='')
        print('still second line')
        print(f'{vars(obj)=}')
        print('waiting two seconds')
        sleep(2)
        print('multiple lines coming up', 'another line', 'third one',
              sep='\n')
        sleep(1)
        print('multi\nline\nstring')
        sleep(1)
        obj.prev_timestamp = None
        print('Print full timestamp on a long line.  ' * 5, end='', flush=True)
        sleep(1)
        print('finish long line')
        pp = ProgressPrinter('looping over range...')
        for i in pp(range(1000)):
            sleep(0.03)
        print('loop all done, sleeping for 2s')
        sleep(2)
        print('ALL DONE')
