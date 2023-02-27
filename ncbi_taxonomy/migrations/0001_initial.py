# Generated by Django 3.2.16 on 2023-02-27 15:20

from django.db import migrations, models
import django.db.models.deletion
import django.db.models.manager
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
            managers=[
                ('loader', django.db.models.manager.Manager()),
            ],
        ),
        migrations.CreateModel(
            name='Division',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('division_id', models.PositiveIntegerField(help_text='taxonomy database division id', unique=True)),
                ('cde', models.CharField(help_text='GenBank division code (three characters)', max_length=3)),
                ('name', models.TextField(unique=True)),
                ('comments', models.TextField(blank=True, default='')),
            ],
            options={
                'abstract': False,
            },
            managers=[
                ('loader', django.db.models.manager.Manager()),
            ],
        ),
        migrations.CreateModel(
            name='Gencode',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('genetic_code_id', models.PositiveIntegerField(help_text='GenBank genetic code id', unique=True)),
                ('abbreviation', models.CharField(blank=True, default='', help_text='genetic code name abbreviation', max_length=10)),
                ('name', models.TextField(help_text='genetic code name', unique=True)),
                ('cde', models.TextField(blank=True, default='', help_text='translation table for this genetic code')),
                ('starts', models.TextField(blank=True, default='', help_text='start codons for this genetic code')),
            ],
            options={
                'abstract': False,
            },
            managers=[
                ('loader', django.db.models.manager.Manager()),
            ],
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
                ('comments', models.TextField(blank=True, default=None, null=True)),
                ('is_pgc_inherited', models.BooleanField(blank=True, default=None, help_text='node inherits plastid gencode from parent', null=True)),
                ('has_specified_species', models.BooleanField(help_text="species in the node's lineage has formal name")),
                ('is_hgc_inherited', models.BooleanField(help_text='inherits hydrogenosome gencode from parent')),
                ('division', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ncbi_taxonomy.division')),
                ('gencode', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='node', to='ncbi_taxonomy.gencode')),
                ('hydro_gencode', models.ForeignKey(blank=True, default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='node_hydro', to='ncbi_taxonomy.gencode')),
                ('mito_gencode', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='node_mito', to='ncbi_taxonomy.gencode')),
                ('parent', models.ForeignKey(blank=True, default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='children', to='ncbi_taxonomy.taxnode')),
                ('plastid_gencode', models.ForeignKey(blank=True, default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='node_plastid', to='ncbi_taxonomy.gencode')),
            ],
            options={
                'verbose_name': 'NCBI taxon',
                'verbose_name_plural': 'NCBI taxa',
            },
            managers=[
                ('loader', django.db.models.manager.Manager()),
            ],
        ),
        migrations.CreateModel(
            name='TypeMaterialType',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(help_text='name of type material type', max_length=64, unique=True)),
                ('synonyms', models.CharField(help_text='alternative names for type material type', max_length=128)),
                ('nomenclature', models.CharField(help_text='Taxonomic Code of Nomenclature coded by a single letter', max_length=2)),
                ('description', models.TextField(help_text='descriptive text')),
            ],
            options={
                'abstract': False,
            },
            managers=[
                ('loader', django.db.models.manager.Manager()),
            ],
        ),
        migrations.CreateModel(
            name='TypeMaterial',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('tax_name', models.TextField(help_text='organism name type material is assigned to')),
                ('identifier', models.CharField(help_text='identifier in type material collection', max_length=32)),
                ('material_type', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ncbi_taxonomy.typematerialtype')),
                ('node', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ncbi_taxonomy.taxnode')),
            ],
            options={
                'abstract': False,
            },
            managers=[
                ('loader', django.db.models.manager.Manager()),
            ],
        ),
        migrations.CreateModel(
            name='TaxName',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('name', models.TextField(help_text='the name itself')),
                ('unique_name', models.TextField(help_text='the unique variant of this name if name not unique')),
                ('name_class', models.PositiveSmallIntegerField(choices=[(1, 'acronym'), (2, 'authority'), (3, 'blast name'), (4, 'common name'), (5, 'equivalent name'), (6, 'genbank acronym'), (7, 'genbank common name'), (8, 'genbank synonym'), (9, 'in-part'), (10, 'includes'), (11, 'scientific name'), (12, 'synonym')], help_text='synonym, common name, ...')),
                ('node', models.ForeignKey(help_text='the node associated with this name', on_delete=django.db.models.deletion.CASCADE, to='ncbi_taxonomy.taxnode')),
            ],
            managers=[
                ('loader', django.db.models.manager.Manager()),
            ],
        ),
        migrations.CreateModel(
            name='MergedNodes',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('old_taxid', models.PositiveIntegerField(help_text='id of nodes which has been merged', unique=True)),
                ('new_node', models.ForeignKey(help_text='node which is result of merging', on_delete=django.db.models.deletion.CASCADE, to='ncbi_taxonomy.taxnode')),
            ],
            options={
                'abstract': False,
            },
            managers=[
                ('loader', django.db.models.manager.Manager()),
            ],
        ),
        migrations.CreateModel(
            name='Host',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('potential_hosts', models.TextField(help_text="theoretical host list separated by comma ','")),
                ('node', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='ncbi_taxonomy.taxnode')),
            ],
            managers=[
                ('loader', django.db.models.manager.Manager()),
            ],
        ),
        migrations.CreateModel(
            name='Citation',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('cit_id', models.PositiveIntegerField(help_text='the unique id of citation', unique=True)),
                ('cit_key', models.TextField(help_text='citation key')),
                ('medline_id', models.PositiveIntegerField(blank=True, default=None, help_text='unique id in MedLine database', null=True)),
                ('pubmed_id', models.PositiveIntegerField(blank=True, default=None, help_text='unique id in PubMed database', null=True)),
                ('url', models.TextField(help_text='URL associated with citation')),
                ('text', models.TextField(help_text='any text (usually article name and authors)\n            The following characters are escaped in this text by a backslash\n            newline (appear as "\n"),\n            tab character ("\t"),\n            double quotes (\'"\'),\n            backslash character ("\\").\n        ')),
                ('node', models.ManyToManyField(to='ncbi_taxonomy.TaxNode')),
            ],
            options={
                'abstract': False,
            },
            managers=[
                ('loader', django.db.models.manager.Manager()),
            ],
        ),
        migrations.AddIndex(
            model_name='taxname',
            index=models.Index(fields=['node', 'name_class'], name='ncbi_taxono_node_id_44c7f5_idx'),
        ),
        migrations.AlterUniqueTogether(
            name='taxname',
            unique_together={('node', 'name', 'name_class')},
        ),
        migrations.AlterUniqueTogether(
            name='host',
            unique_together={('node', 'potential_hosts')},
        ),
    ]
