# Generated by Django 3.2.18 on 2023-03-15 16:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('omics', '0002_auto_20230227_1431'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sample',
            name='sample_id',
            field=models.CharField(default='bad_sample_id', help_text='internal sample accession', max_length=32, unique=True),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='sample',
            name='sample_name',
            field=models.TextField(blank=True, default='', help_text='sample ID or name as given by study', max_length=32),
        ),
        migrations.AlterField(
            model_name='sample',
            name='sra_accession',
            field=models.TextField(blank=True, default='', max_length=16, verbose_name='SRA accession'),
        ),
    ]
