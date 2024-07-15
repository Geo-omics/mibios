"""
glamr specific data loading jobs

To register these jobs, run
omics.tracking.registry.register_from_module(module_name) in AppConfig.ready()
"""
from mibios.omics.jobs import RegisterWithPipeline
from mibios.omics.models import SampleTracking
from mibios.omics.tracking import Job


class LoadMetaData(Job):
    flag = SampleTracking.Flag.METADATA
    before = [RegisterWithPipeline]
