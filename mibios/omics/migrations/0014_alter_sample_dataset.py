# Generated by Django 3.2.19 on 2024-05-22 20:39

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.OMICS_DATASET_MODEL),
        ('omics', '0013_alter_sample_sample_type'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sample',
            name='dataset',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='glamr.dataset'),
        ),
    ]