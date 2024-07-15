from . import get_sample_model
from .models import File, SampleTracking
from .tracking import Job


class RegisterWithPipeline(Job):
    flag = SampleTracking.Flag.PIPELINE


class LoadMetaGAssembly(Job):
    flag = SampleTracking.Flag.ASSEMBLY
    after = ['RegisterWithPipeline']
    sample_types = [get_sample_model().TYPE_METAGENOME]
    requires_files = [File.Type.METAG_ASM]


class LoadUR1Abund(Job):
    flag = SampleTracking.Flag.UR1ABUND
    after = ['RegisterWithPipeline']
    sample_types = [get_sample_model().TYPE_METAGENOME]
    requires_files = [File.Type.FUNC_ABUND]


class LoadTaxAbund(Job):
    flag = SampleTracking.Flag.TAXABUND
    after = ['RegisterWithPipeline']
    sample_types = [get_sample_model().TYPE_METAGENOME]
    requires_files = [File.Type.TAX_ABUND]
