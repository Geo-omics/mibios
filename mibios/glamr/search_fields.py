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
        'taxname': ['name'],  # what about 'unique_name' ?
    },
    'umrad': {
        'compoundname': ['entry'],
        'functionname': ['entry'],
        'uniref100': ['accession', 'uniref90'],
    },
}


def print():
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
