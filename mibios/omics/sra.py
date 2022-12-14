""" helper library for SRA access """
from pathlib import Path
import shutil
import subprocess
import tempfile

from Bio import Entrez

import defusedxml.ElementTree as ET


Entrez.email = 'heinro@umich.edu'
Entrez.tool = 'glamr'


def get_sample(accession):
    """ retrieve SRA sample entry from accession """
    h = Entrez.efetch(db='sra', id=accession)
    root = ET.fromstring(h.read().decode())
    h.close()

    return expand_element(root)


def search(accession):
    """ search SRA  """
    h = Entrez.esearch(db='sra', term=accession)
    res = Entrez.read(h)

    try:
        id_list = res['IdList']
    except KeyError as e:
        raise RuntimeError(f'expected id list in result: {res}') from e

    h = Entrez.efetch(db='sra', id=','.join(id_list))
    root = ET.fromstring(h.read().decode())
    h.close()
    return expand_element(root)


def get_run(biosample, other=None):
    """ get run+platform info  from biosample and other SRA accession """
    LAYOUT_PAIRED = 'PAIRED'
    LAYOUT_SINGLE = 'SINGLE'

    if not biosample.startswith('SAM'):
        raise ValueError(
            'expecting biosample accessions to have SAM prefix'
        )

    have_srr = False
    have_srx = False
    if other is not None:
        if other.startswith('SRR'):
            have_srr = True
        elif other.startswith('SRX'):
            have_srx = True
        else:
            raise ValueError('other must be SRR or SRX accession')

    finds = []
    res = search(biosample)
    if isinstance(res, dict):
        # TODO: is it always a dict? can it be a list?
        res = [res]
    for xp in res:
        pkg_items = xp['EXPERIMENT_PACKAGE']
        if isinstance(pkg_items, dict):
            pkg_items = [pkg_items]
        for i in pkg_items:
            runs = i['RUN_SET']
            if isinstance(runs, dict):
                runs = [runs]
            # platform: should be a single key dict of a dict, take outer key
            # e.g.: 'PLATFORM': {'LS454': {'INSTRUMENT_MODEL': {}}}
            #       and extract 'LS454'
            platform = list(i['EXPERIMENT']['PLATFORM'].keys())[0]
            # layout: see above, similar to platform
            layout = list(i['EXPERIMENT']['DESIGN']['LIBRARY_DESCRIPTOR']['LIBRARY_LAYOUT'].keys())[0]  # noqa:E501
            if layout not in [LAYOUT_PAIRED, LAYOUT_SINGLE]:
                print(f'WARNING: unknown layout: {layout}')
            is_paired_end = layout == LAYOUT_PAIRED
            all_runs = []
            for j in runs:
                run = j['RUN']
                run_id = run['accession']
                exp_id = run['EXPERIMENT_REF']['accession']
                if have_srr and run_id == other:
                    return run, platform, is_paired_end
                elif have_srx and exp_id == other:
                    return run, platform, is_paired_end
                all_runs.append(run)
            sample = i['SAMPLE']
            finds.append((
                (sample['alias'], sample['accession'], platform, layout),
                all_runs,
            ))

    if have_srr or have_srx:
        raise RuntimeError(f'Not found. Only got these: {finds}')

    if len(finds) == 1:
        if len(finds[0][1]) == 1:
            # run set has single run, return it+platform+layout
            is_paired_end = finds[0][0][3] == LAYOUT_PAIRED
            return finds[0][1][0], finds[0][0][2], is_paired_end
        else:
            raise RuntimeError(f'zero or multiple runs in runset: {finds}')
    elif len(finds) > 1:
        raise RuntimeError(
            f'found more than one run, please give run or experiment '
            f'accession: {finds}'
        )
    else:
        raise RuntimeError('no runs found at all(?): {finds}')


def expand_element(elem, always_keep_lists=False):
    """ expand xml element into dict-of-dicts """
    data = dict(**elem.attrib)
    for i in list(elem):
        key = i.tag
        if key in data:
            # FIXME? just keeep lists
            if isinstance(data[key], dict):
                # convert into list
                data[key] = [data[key]]
            data[key].append(expand_element(i))
        else:
            data[key] = expand_element(i)
    return data


def download_fastq(run_accession, dest=None, verbose=False):
    """
    get fastq file(s) with fastq-dump tool

    dest: pathlib.Path of destination file(s) without .fasta suffix
    Returns a list of downloaded files.

    Existing files will be overwritten.
    """
    if not run_accession.startswith('SRR'):
        raise ValueError('expecting an SRA run accession (SRRxxxxxxx)')

    if dest is None:
        dest = Path() / run_accession

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp = Path(tmpdirname)
        cmd = [
            'fasterq-dump',
            '--outfile', f'{tmp / dest.name}.fastq',
            '--temp', '/tmp',
            '--split-3',
            '--skip-technical',
            '--verbose',
            '--details',
            '--log-level', 'info',
            run_accession,
        ]
        if verbose:
            print('Running:', *cmd)

        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # move from tmp to destination
        files = []
        dest.parent.mkdir(parents=True, exist_ok=True)
        for i in tmp.iterdir():
            to = dest.parent / i.name
            shutil.move(i, to)
            files.append(to)

    with (dest.parent / 'fasterq-dump.log').open('ab') as logfile:
        logfile.write(p.stdout)

    if verbose:
        print('got', len(files), 'file(s), log appended to',
              logfile.name)

    if p.returncode:
        print(p.stdout.decode())
        raise RuntimeError(
            f'call to fasterq-dump failed, check log file, '
            f'returncode={p.returncode}'
        )

    return files
