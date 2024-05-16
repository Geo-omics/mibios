from django.apps import apps
from django.utils.functional import lazy

from mibios import get_registry


SEARCH_FIELDS = {
    'glamr': {
        'dataset': [
            'bioproject',
            'jgi_project',
            'gold_id',
            'scheme',
            'material_type',
            'water_bodies',
            'primers',
            'sequencing_target',
            'sequencing_platform',
            # 'note' ?
        ],
        'sample': [
            'sample_name',
            'sample_type',
            'sra_accession',
            'amplicon_target',
            'fwd_primer',
            'rev_primer',
            'project_id',
            'biosample',
            'geo_loc_name',
            'gaz_id',
            'noaa_site',
            'env_broad_scale',
            'env_local_scale',
            'env_medium',
            'keywords',
        ],
        'reference': [
            'short_reference',
            'authors',
            'title',
            'abstract',
            'key_words',
            'publication',
            'doi',
        ],
    },
    'ncbi_taxonomy': {
        'taxnode': ['taxname__name'],
    },
    'umrad': {
        # 'compoundname': ['entry'],
        'functionname': ['entry'],
    },
}


ADVANCED_SEARCH_MODELS = [
    'dataset', 'sample', 'reference', 'compoundname', 'contig',
    'functionname', 'gene', 'taxname',
]
""" names of models we offer for the advanced search """


def compile_search_fields():
    """
    Process the SEARCH_FIELDS for use by the rest of the module

    The compiled dict maps models to list of fields.
    """
    retval = {}
    for app_label, data in SEARCH_FIELDS.items():
        appconf = apps.get_app_config(app_label)
        for model_name, field_list in data.items():
            model = appconf.get_model(model_name)
            retval[model] = field_list

    return retval


# this must be evaluated lazily to ensure apps and models are set up
search_fields = lazy(compile_search_fields, dict)()


def printit():
    """ helper go generate SEARCH_FIELD """
    r = get_registry()
    d = {}
    for a in r.apps:
        d[a] = {}
        for m in r.get_models(a):
            d[a][m._meta.model_name] = []
            for f in m._meta.get_fields():
                d[a][m._meta.model_name].append(f.name)
    print(d)
