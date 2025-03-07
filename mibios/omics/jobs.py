from . import get_sample_model
from .models import Bin, Contig, File, ReadAbundance, SampleTracking, \
    TaxonAbundance
from .tracking import Job


Sample = get_sample_model()


class RegisterWithPipeline(Job):
    flag = SampleTracking.Flag.PIPELINE


class LoadMetaGAssembly(Job):
    flag = SampleTracking.Flag.ASSEMBLY
    after = [RegisterWithPipeline]
    sample_types = [Sample.TYPE_METAGENOME]
    required_files = [File.Type.METAG_ASM]
    run = Contig.loader.load_fasta
    undo = Contig.loader.unload_fasta


class LoadContigAbund(Job):
    flag = SampleTracking.Flag.CABUND
    after = [LoadMetaGAssembly]
    sample_types = [Sample.TYPE_METAGENOME]
    required_files = [File.Type.CONT_ABUND]
    run = Contig.loader.load_abundance
    undo = Contig.loader.unload_abundance


class LoadContigLCA(Job):
    # FIXME: do we still want this data in the DB?
    enabled = False
    flag = ...
    after = [LoadMetaGAssembly]
    sample_types = [Sample.TYPE_METAGENOME]
    required_files = ...  # *_contig_lca.tsv
    run = Contig.loader.load_lca


class LoadUR1Abund(Job):
    flag = SampleTracking.Flag.UR1ABUND
    after = [RegisterWithPipeline]
    sample_types = [Sample.TYPE_METAGENOME]
    required_files = [File.Type.FUNC_ABUND]
    run = ReadAbundance.loader.load_sample
    undo = ReadAbundance.loader.unload_sample


class LoadUR1TPM(Job):
    flag = SampleTracking.Flag.UR1TPM
    after = [LoadUR1Abund]
    sample_types = [Sample.TYPE_METAGENOME]
    required_files = [File.Type.FUNC_ABUND_TPM]
    run = ReadAbundance.loader.load_tpm_sample
    undo = ReadAbundance.loader.unload_tpm_sample


class LoadTaxAbund(Job):
    flag = SampleTracking.Flag.TAXABUND
    after = [RegisterWithPipeline]
    sample_types = [Sample.TYPE_METAGENOME]
    required_files = [File.Type.TAX_ABUND]
    run = TaxonAbundance.loader.load_sample


class LoadBins(Job):
    flag = SampleTracking.Flag.BINNING
    after = [LoadMetaGAssembly]
    sample_types = [Sample.TYPE_METAGENOME]
    required_files = [
        File.Type.BIN_COV,
        File.Type.BIN_CLASS_ARC,
        File.Type.BIN_CLASS_BAC,
        File.Type.BIN_CHECKM,
        File.Type.BIN_CONTIG,
    ]
    run = Bin.loader.load_sample
    undo = Bin.loader.unload_sample
