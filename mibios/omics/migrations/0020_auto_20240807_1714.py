# Generated by Django 3.2.19 on 2024-08-07 21:14

from django.db import migrations, models
import mibios.omics.models
import mibios.umrad.fields


class Migration(migrations.Migration):

    dependencies = [
        ('omics', '0019_auto_20240719_1529'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sample',
            name='sample_name',
            field=models.TextField(blank=True, default='', help_text='sample ID or name as given by original data source', max_length=32),
        ),
        migrations.AlterField(
            model_name='sample',
            name='sra_accession',
            field=models.TextField(blank=True, default='', max_length=16),
        ),
    ]
