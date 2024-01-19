# Generated by Django 4.2.6 on 2024-01-19 16:15

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.OMICS_SAMPLE_MODEL),
        ("omics", "0011_auto_20231110_1144"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="readabundance",
            options={
                "default_manager_name": "objects",
                "verbose_name": "functional abundance",
            },
        ),
        migrations.AlterField(
            model_name="readabundance",
            name="sample",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="functional_abundance",
                to=settings.OMICS_SAMPLE_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="taxonabundance",
            name="tpm",
            field=models.FloatField(verbose_name="TPM"),
        ),
    ]
