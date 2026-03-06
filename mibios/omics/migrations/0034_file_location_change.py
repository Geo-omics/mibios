from django.db import migrations
from django.db.models import F, Value
from django.db.models.functions import Concat, Substr


def forward(apps, schema_editor):
    """
    Add omics/ to every non-empty path.  Expecting all current files to be
    under omics/
    """
    File = apps.get_model('omics.File')
    fields = ('file_pipeline', 'file_local', 'file_globus')
    # (a) Follow change of OMICS_PIPELINE_DATA to its parent, file remain same
    # on pipeline storage filesystem
    # (b) On local and globus storage filesystem 'omics' directory must be
    # manually created and files moved
    for field in fields:
        qs = File.objects.exclude(**{field: ''})
        qs.update(**{field: Concat(Value('omics/'), F(field))})


def reverse(apps, schema_editor):
    """
    Make every path relative to omics
    """
    File = apps.get_model('omics.File')
    fields = ('file_pipeline', 'file_local', 'file_globus')
    for field in fields:
        qs = File.objects.filter(**{field + '__startswith': 'omics/'})
        qs.update(**{field: Substr(F(field), len('omics/') + 1)})
        qs = File.objects.filter(**{field + '__startswith': 'projects/'})
        qs.update(**{field: Concat(Value('../'), F(field))})


class Migration(migrations.Migration):

    dependencies = [
        ('omics', '0033_datasettracking_and_more'),
    ]

    operations = [
        migrations.RunPython(forward, reverse)
    ]
