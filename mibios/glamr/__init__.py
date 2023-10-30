"""
sub-package for the GLAMR DB and website

glamr depends on omics, ncbi_taxonomy, and umrad apps.

Populating the database:

    from mibios.omics.models import *
    from mibios.glamr.models import *
    Sample.loader.load_all_meta_data()

    from mibios.ncbi_taxonomy import load()
    load()

    from mibios.umrad.models import *
    CompoundRecord.loader.load(skip_on_error=True)
    ReactionRecord.loader.load(skip_on_error=True)
    UniRef100.loader.load(skip_on_error=True)
    FuncRefDBEntry.name_loader.load()

    Sample.loader.load_metagenomic_data()


"""

GREAT_LAKES = [
    'Lake Erie',
    'Lake Huron',
    'Lake Michigan',
    'Lake Ontario',
    'Lake Superior',
]
