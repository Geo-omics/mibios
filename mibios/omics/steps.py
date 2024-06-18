from . import get_sample_model
from .models import File, SampleTracking
from .tracking import Step


class RegisterWithPipeline(Step):
    flag = SampleTracking.Flag.PIPELINE


class LoadMetaGAssembly(Step):
    flag = SampleTracking.Flag.ASSEMBLY
    after = ['RegisterWithPipeline']
    sample_types = [get_sample_model().TYPE_METAGENOME]
    requires_files = [File.Type.METAG_ASM]


class LoadUR1Abund(Step):
    flag = SampleTracking.Flag.UR1ABUND
    after = ['RegisterWithPipeline']
    sample_types = [get_sample_model().TYPE_METAGENOME]
    requires_files = [File.Type.FUNC_ABUND]


class LoadTaxAbund(Step):
    flag = SampleTracking.Flag.TAXABUND
    after = ['RegisterWithPipeline']
    sample_types = [get_sample_model().TYPE_METAGENOME]
    requires_files = [File.Type.TAX_ABUND]
