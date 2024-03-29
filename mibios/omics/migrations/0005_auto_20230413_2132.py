# Generated by Django 3.2.18 on 2023-04-14 01:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('omics', '0004_auto_20230410_1653'),
    ]

    operations = [
        migrations.AlterField(
            model_name='contig',
            name='contig_id',
            field=models.TextField(max_length=32),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='short_name',
            field=models.TextField(blank=True, default=None, help_text='a short name or description, for internal use, not (necessarily) for public display', max_length=128, null=True, unique=True),
        ),
        migrations.AlterField(
            model_name='gene',
            name='gene_id',
            field=models.TextField(max_length=32),
        ),
    ]
