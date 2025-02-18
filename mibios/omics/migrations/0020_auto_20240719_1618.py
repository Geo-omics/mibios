# Generated by Django 3.2.19 on 2024-07-19 20:18

from django.db import migrations
import mibios.omics.models
import mibios.umrad.fields


try:
    file_path_root = mibios.omics.models.File.get_path_prefix
except AttributeError:
    file_path_root = None

try:
    file_public_root = mibios.omics.models.File.get_public_prefix
except AttributeError:
    file_public_root = None


class Migration(migrations.Migration):

    dependencies = [
        ('omics', '0019_auto_20240719_1529'),
    ]

    operations = [
        migrations.AlterField(
            model_name='file',
            name='path',
            field=mibios.umrad.fields.PathField(root=file_path_root, unique=True),
        ),
        migrations.AlterField(
            model_name='file',
            name='public',
            field=mibios.umrad.fields.PathField(blank=True, null=True, root=file_public_root),
        ),
    ]
