from .models import (
    ASVAbundance, Bin, Contig, File, ReadAbundance, DataTracking, SeqSample,
    TaxonAbundance,
)
from .tracking import BaseJob, DatasetJob, SeqSampleJob


class RegisterWithPipeline(BaseJob):
    flag = DataTracking.Flag.PIPELINE


class LoadMetaGAssembly(SeqSampleJob):
    flag = DataTracking.Flag.ASSEMBLY
    after = [RegisterWithPipeline]
    sample_types = [SeqSample.Type.METAGENOME]
    required_files = [File.Type.METAG_ASM]
    run = Contig.loader.load_fasta
    undo = Contig.loader.unload_fasta


class LoadContigAbund(SeqSampleJob):
    flag = DataTracking.Flag.CABUND
    after = [LoadMetaGAssembly]
    sample_types = [SeqSample.Type.METAGENOME]
    required_files = [File.Type.CONT_ABUND]
    run = Contig.loader.load_abundance
    undo = Contig.loader.unload_abundance


class LoadContigLCA(SeqSampleJob):
    # FIXME: do we still want this data in the DB?
    enabled = False
    flag = ...
    after = [LoadMetaGAssembly]
    sample_types = [SeqSample.Type.METAGENOME]
    required_files = ...  # *_contig_lca.tsv
    run = Contig.loader.load_lca


class LoadUR1Abund(SeqSampleJob):
    flag = DataTracking.Flag.UR1ABUND
    after = [RegisterWithPipeline]
    sample_types = [SeqSample.Type.METAGENOME]
    required_files = [File.Type.FUNC_ABUND]
    run = ReadAbundance.loader.load_sample
    undo = ReadAbundance.loader.unload_sample


class LoadUR1TPM(SeqSampleJob):
    flag = DataTracking.Flag.UR1TPM
    after = [LoadUR1Abund]
    sample_types = [SeqSample.Type.METAGENOME]
    required_files = [File.Type.FUNC_ABUND_TPM]
    run = ReadAbundance.loader.load_tpm_sample
    undo = ReadAbundance.loader.unload_tpm_sample


class LoadTaxAbund(SeqSampleJob):
    flag = DataTracking.Flag.TAXABUND
    after = [RegisterWithPipeline]
    sample_types = [SeqSample.Type.METAGENOME]
    required_files = [File.Type.TAX_ABUND]
    run = TaxonAbundance.loader.load_sample


class LoadBins(SeqSampleJob):
    flag = DataTracking.Flag.BINNING
    after = [LoadMetaGAssembly]
    sample_types = [SeqSample.Type.METAGENOME]
    required_files = [
        File.Type.BIN_COV,
        File.Type.BIN_CLASS_ARC,
        File.Type.BIN_CLASS_BAC,
        File.Type.BIN_CHECKM,
        File.Type.BIN_CONTIG,
    ]
    run = Bin.loader.load_sample
    undo = Bin.loader.unload_sample


class LoadASVAbund(DatasetJob):
    flag = DataTracking.Flag.ASVABUND
    after = [RegisterWithPipeline]
    sample_types = [SeqSample.Type.AMPLICON]
    required_files = [
        File.Type.DADA_ASV,
        File.Type.DADA_ABUND,
    ]
    run = ASVAbundance.loader.load_dataset
    undo = ASVAbundance.loader.unload_dataset

    def get_params(self):
        """
        Spawn one subjob per amplicon target
        """
        return [
            # cf. FileType.DADA_ASV etc
            {'amplicon_target': target}
            for _, target
            in self.subject.get_amplicon_pipeline_results().keys()
        ]
