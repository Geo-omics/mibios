# Generated by Django 2.2.24 on 2021-12-15 19:28

from django.db import migrations, models
import django.db.models.deletion
import mibios.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DeletedNode',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('taxid', models.PositiveIntegerField(help_text='deleted node id', unique=True)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Division',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('division_id', models.PositiveIntegerField(help_text='taxonomy database division id', unique=True)),
                ('cde', models.CharField(help_text='GenBank division code (three characters)', max_length=3)),
                ('name', models.CharField(max_length=3)),
                ('comments', models.CharField(max_length=1000)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Gencode',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('genetic_code_id', models.PositiveIntegerField(help_text='GenBank genetic code id', unique=True)),
                ('abbreviation', models.CharField(help_text='genetic code name abbreviation', max_length=10)),
                ('name', models.CharField(help_text='genetic code name', max_length=64, unique=True)),
                ('cde', models.CharField(help_text='translation table for this genetic code', max_length=64)),
                ('starts', models.CharField(help_text='start codons for this genetic code', max_length=64)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='TaxNode',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('taxid', models.PositiveIntegerField(unique=True, verbose_name='taxonomy ID')),
                ('rank', models.CharField(db_index=True, max_length=32)),
                ('embl_code', models.CharField(blank=True, default='', max_length=2)),
                ('is_div_inherited', models.BooleanField()),
                ('is_gencode_inherited', models.BooleanField()),
                ('is_mgc_inherited', models.BooleanField(help_text='node inherits mitochondrial gencode from parent')),
                ('is_genbank_hidden', models.BooleanField(help_text='name is suppressed in GenBank entry lineage')),
                ('hidden_subtree_root', models.BooleanField(help_text='this subtree has no sequence data yet')),
                ('comments', models.CharField(max_length=1000)),
                ('is_pgc_inherited', models.NullBooleanField(default=None)),
                ('has_specified_species', models.BooleanField()),
                ('is_hgc_inherited', models.BooleanField()),
                ('division', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ncbi_taxonomy.Division')),
                ('gencode', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='node', to='ncbi_taxonomy.Gencode')),
                ('hydro_gencode', models.ForeignKey(blank=True, default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='node_hydro', to='ncbi_taxonomy.Gencode')),
                ('mito_gencode', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='node_mito', to='ncbi_taxonomy.Gencode')),
                ('parent', models.ForeignKey(blank=True, default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, to='ncbi_taxonomy.TaxNode')),
                ('plastid_gencode', models.ForeignKey(blank=True, default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='node_plastid', to='ncbi_taxonomy.Gencode')),
            ],
            options={
                'verbose_name': 'NCBI taxon',
                'verbose_name_plural': 'ncbi taxa',
            },
        ),
        migrations.CreateModel(
            name='TypeMaterialType',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(help_text='name of type material type', max_length=64, unique=True)),
                ('synonyms', models.CharField(help_text='alternative names for type material type', max_length=128)),
                ('nomenclature', models.CharField(help_text='Taxonomic Code of Nomenclature coded by a single letter', max_length=2)),
                ('description', models.CharField(help_text='descriptive text', max_length=1024)),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='TypeMaterial',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('tax_name', models.CharField(help_text='organism name type material is assigned to', max_length=128)),
                ('identifier', models.CharField(help_text='identifier in type material collection', max_length=32)),
                ('material_type', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ncbi_taxonomy.TypeMaterialType')),
                ('node', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ncbi_taxonomy.TaxNode')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='MergedNodes',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('old_taxid', models.PositiveIntegerField(help_text='id of nodes which has been merged', unique=True)),
                ('new_node', models.ForeignKey(help_text='node which is result of merging', on_delete=django.db.models.deletion.CASCADE, to='ncbi_taxonomy.TaxNode')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Host',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('potential_hosts', models.CharField(help_text="theoretical host list separated by comma ','", max_length=32)),
                ('node', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ncbi_taxonomy.TaxNode')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Citation',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('cit_id', models.PositiveIntegerField(help_text='the unique id of citation', unique=True)),
                ('cit_key', models.CharField(help_text='citation key', max_length=512)),
                ('medline_id', models.PositiveIntegerField(blank=True, help_text='unique id in MedLine database', null=True)),
                ('pubmed_id', models.PositiveIntegerField(blank=True, help_text='unique id in PubMed database', null=True)),
                ('url', models.CharField(help_text='URL associated with citation', max_length=512)),
                ('text', models.CharField(help_text='any text (usually article name and authors)\n            The following characters are escaped in this text by a backslash\n            newline (appear as "\n"),\n            tab character ("\t"),\n            double quotes (\'"\'),\n            backslash character ("\\").\n        ', max_length=1024)),
                ('node', models.ManyToManyField(to='ncbi_taxonomy.TaxNode')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='TaxName',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(help_text='the name itself', max_length=128)),
                ('unique_name', models.CharField(help_text='the unique variant of this name if name not unique', max_length=128)),
                ('name_class', models.CharField(db_index=True, help_text='synonym, common name, ...', max_length=32)),
                ('node', models.ForeignKey(help_text='the node associated with this name', on_delete=django.db.models.deletion.CASCADE, to='ncbi_taxonomy.TaxNode')),
            ],
            options={
                'unique_together': {('node', 'name', 'name_class')},
            },
        ),
    ]
