from time import sleep

from mibios.omics.utils import Timestamper
from mibios.umrad.utils import ProgressPrinter


def test_timestamper():
    with Timestamper(template='[ {timestamp} testing ]  ') as obj:
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
        pp = ProgressPrinter('looping over range...')
        for i in pp(range(1000)):
            sleep(0.03)
        print('loop all done, sleeping for 2s')
        sleep(2)
        print('ALL DONE')
