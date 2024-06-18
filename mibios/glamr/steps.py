"""
glamr specific data loading steps

To register these steps, run
omics.tracking.registry.register_from_module(module_name) in AppConfig.ready()
"""
from mibios.omics.steps import RegisterWithPipeline
from mibios.omics.tracking import SampleTracking, Step


class LoadMetaData(Step):
    flag = SampleTracking.Flag.METADATA
    before = [RegisterWithPipeline]
