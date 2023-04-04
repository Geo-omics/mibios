# Generated by Django 3.2.18 on 2023-03-21 18:48

import django.contrib.postgres.indexes
from django.contrib.postgres.operations import TrigramExtension
import django.contrib.postgres.search
from django.db import migrations, models
import django.db.models.deletion

from . import AddTSVectorField


class Migration(migrations.Migration):

    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
        ('glamr', '0003_auto_20230315_1213'),
    ]

    operations = [
        migrations.DeleteModel(
            name='SearchTerm',
        ),
        migrations.CreateModel(
            name='Searchable',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField(max_length=32)),
                ('has_hit', models.BooleanField(default=False)),
                ('field', models.CharField(max_length=100)),
                ('object_id', models.PositiveIntegerField()),
            ],
        ),
        AddTSVectorField(
            model_name='searchable',
            name='searchvector',
            field=django.contrib.postgres.search.SearchVectorField(null=True),
            target_field_name='text',
        ),
        migrations.CreateModel(
            name='UniqueWord',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('word', models.TextField()),
            ],
        ),
        TrigramExtension(),
        migrations.AddIndex(
            model_name='uniqueword',
            index=django.contrib.postgres.indexes.GistIndex(fields=['word'], name='term_trigram_idx', opclasses=['gist_trgm_ops']),
        ),
        migrations.AddField(
            model_name='searchable',
            name='content_type',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='contenttypes.contenttype'),
        ),
        migrations.AddIndex(
            model_name='searchable',
            index=django.contrib.postgres.indexes.GinIndex(fields=['searchvector'], name='glamr_searc_searchv_f71dcf_gin'),
        ),
    ]