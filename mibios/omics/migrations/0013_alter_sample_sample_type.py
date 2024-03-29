# Generated by Django 3.2.19 on 2024-02-09 20:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('omics', '0012_alter_readabundance_options_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sample',
            name='sample_type',
            field=models.CharField(blank=True, choices=[('amplicon', 'amplicon'), ('metagenome', 'metagenome'), ('metatranscriptome', 'metatranscriptome'), ('isolate_genome', 'isolate_genome')], default=None, max_length=32, null=True),
        ),
    ]
