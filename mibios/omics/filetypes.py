from pathlib import Path

from django.db.models import IntegerChoices


class FileType(IntegerChoices):
    """
    Enumeration for use in Field(choices=...) in the File model.  To replace
    IntgergerChoices but with extra attributes.

    Each instance gets passed a tuple of integer value and a dict.  The dict is
    required to define a "label" similar to IntegerChoices and additionally a
    "path" that sets a template string of the file path relative to the
    analysis directory.  An optional boolean "with_dataset", if True, declares
    to use the dataset analysis directory, and if False (the default) to use
    the sample analysis directory as base.

    Path templates support the following parameters:
        sample
        dataset
        amplicon_target
    """
    METAG_ASM = (1, dict(
        label='metagenomic assembly, fasta format',
        path=Path(
            'assembly', 'megahit_noNORM', 'final.contigs.renamed.fa'
        ),
    ))
    FUNC_ABUND = (3, dict(
        label='functional abundance, csv format',
        path='{sample.sample_id}_tophit_report',
    ))
    TAX_ABUND = (4, dict(
        label='taxonomic abundance, csv format',
        path='{sample.sample_id}_lca_abund_summarized.tsv',
    ))
    FUNC_ABUND_TPM = (5, dict(
        label='functional abundance (TPM) [csv]',
        path='{sample.sample_id}_tophit_TPM.tsv',
    ))
    CONT_ABUND = (6, dict(
        label='contig abundance [csv]',
        path='{sample.sample_id}_contig_abund.tsv',
    ))
    CONT_LCA = (7, dict(
        label='contig taxonomy [csv]',
        path='{sample.sample_id}_contig_lca.tsv',
    ))
    BIN_COV = (8, dict(
        label='bin abundance',
        path='bins/coverage_drep_bins.tsv',
    ))
    BIN_CLASS_ARC = (9, dict(
        label='GTDB Archaea bin classification',
        path='bins/GTDB/gtdbtk.ar53.summary.tsv',
    ))
    BIN_CLASS_BAC = (10, dict(
        label='GTDB Bacteria bin classification',
        path='bins/GTDB/gtdbtk.bac120.summary.tsv',
    ))
    BIN_CHECKM = (11, dict(
        label='CheckM results',
        path='bins/all_raw_bins/checkm.txt',
    ))
    BIN_CONTIG = (12, dict(
        label='binning results',
        path='bins/contig_bins.rds',
    ))
    DADA_ASV = (13, dict(
        label='dada2 dataset ASVs, fasta',
        with_dataset=True,
        path='dada2.{amplicon_target.spec}/rep_seqs.fasta',
    ))
    DADA_ABUND = (14, dict(
        label='dada2 dataset abundance, csv',
        with_dataset=True,
        path='dada2.{amplicon_target.spec}/asv_table.tsv',
    ))

    def __new__(cls, value, attrs_dict):
        # Only the int becomes the value, extra attributes are passed via dict
        obj = int.__new__(cls, value)
        obj._value_ = int(obj)
        obj._label_ = attrs_dict.pop('label')
        obj.path = attrs_dict.pop('path')
        obj.with_dataset = attrs_dict.pop('with_dataset', False)
        for attr, value in attrs_dict.items():
            setattr(obj, attr, value)
        return obj
