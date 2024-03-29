# Generated by Django 3.2.20 on 2023-10-19 19:34

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.db.models.manager
import mibios.models


class Migration(migrations.Migration):

    dependencies = [
        ('ncbi_taxonomy', '0002_auto_20230413_1611'),
        ('umrad', '0002_auto_20230227_1431'),
        migrations.swappable_dependency(settings.OMICS_SAMPLE_MODEL),
        ('omics', '0006_auto_20230919_0910'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='gene',
            options={},
        ),
        migrations.AlterModelManagers(
            name='gene',
            managers=[
                ('loader', django.db.models.manager.Manager()),
            ],
        ),
        migrations.RemoveField(
            model_name='sample',
            name='gene_abundance_loaded',
        ),
        migrations.RemoveField(
            model_name='sample',
            name='gene_fasta_loaded',
        ),
        migrations.RemoveField(
            model_name='taxonabundance',
            name='count_contig',
        ),
        migrations.RemoveField(
            model_name='taxonabundance',
            name='count_gene',
        ),
        migrations.RemoveField(
            model_name='taxonabundance',
            name='len_contig',
        ),
        migrations.RemoveField(
            model_name='taxonabundance',
            name='len_gene',
        ),
        migrations.RemoveField(
            model_name='taxonabundance',
            name='mean_fpkm_contig',
        ),
        migrations.RemoveField(
            model_name='taxonabundance',
            name='mean_fpkm_gene',
        ),
        migrations.RemoveField(
            model_name='taxonabundance',
            name='norm_frags_contig',
        ),
        migrations.RemoveField(
            model_name='taxonabundance',
            name='norm_frags_gene',
        ),
        migrations.RemoveField(
            model_name='taxonabundance',
            name='norm_reads_contig',
        ),
        migrations.RemoveField(
            model_name='taxonabundance',
            name='norm_reads_gene',
        ),
        migrations.RemoveField(
            model_name='taxonabundance',
            name='wmedian_fpkm_contig',
        ),
        migrations.RemoveField(
            model_name='taxonabundance',
            name='wmedian_fpkm_gene',
        ),
        migrations.AddField(
            model_name='contig',
            name='contig_no',
            field=models.PositiveIntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='contig',
            name='covered_bases',
            field=models.PositiveIntegerField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name='contig',
            name='mean',
            field=models.FloatField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name='contig',
            name='reads',
            field=models.PositiveIntegerField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name='contig',
            name='reads_per_base',
            field=models.FloatField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name='contig',
            name='tpm',
            field=models.FloatField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name='contig',
            name='trimmed_mean',
            field=models.FloatField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name='contig',
            name='variance',
            field=models.FloatField(blank=True, default=None, null=True),
        ),
        migrations.AddField(
            model_name='gene',
            name='bitscore',
            field=models.PositiveSmallIntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='gene',
            name='gapopen',
            field=models.PositiveSmallIntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='gene',
            name='mismatch',
            field=models.PositiveSmallIntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='gene',
            name='pident',
            field=models.FloatField(default=0.0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='gene',
            name='qend',
            field=models.PositiveSmallIntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='gene',
            name='qstart',
            field=models.PositiveSmallIntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='gene',
            name='ref',
            field=models.ForeignKey(default=None, on_delete=django.db.models.deletion.CASCADE, to='umrad.uniref100'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='gene',
            name='send',
            field=models.PositiveSmallIntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='gene',
            name='sstart',
            field=models.PositiveSmallIntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='sample',
            name='contig_lca_loaded',
            field=models.BooleanField(default=False, help_text='contig LCA data loaded'),
        ),
        migrations.AddField(
            model_name='sample',
            name='gene_alignments_loaded',
            field=models.BooleanField(default=False, help_text='genes loaded via contig_tophit_aln file'),
        ),
        migrations.AddField(
            model_name='taxonabundance',
            name='tpm',
            field=models.FloatField(default=0.0),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='gene',
            name='length',
            field=models.PositiveSmallIntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='taxonabundance',
            name='taxon',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='abundance', to='ncbi_taxonomy.taxnode'),
        ),
        migrations.AlterUniqueTogether(
            name='contig',
            unique_together={('sample', 'contig_no')},
        ),
        migrations.AlterUniqueTogether(
            name='gene',
            unique_together={('sample', 'contig', 'ref', 'qstart', 'qend', 'sstart', 'send')},
        ),
        migrations.CreateModel(
            name='Abundance',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('unique_cov', models.DecimalField(decimal_places=3, max_digits=4)),
                ('target_cov', models.DecimalField(decimal_places=3, max_digits=4)),
                ('avg_ident', models.DecimalField(decimal_places=3, max_digits=4)),
                ('ref', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='umrad.uniref100')),
                ('sample', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.OMICS_SAMPLE_MODEL)),
            ],
            options={
                'abstract': False,
                'default_manager_name': 'objects',
            },
        ),
        migrations.RemoveField(
            model_name='contig',
            name='bases',
        ),
        migrations.RemoveField(
            model_name='contig',
            name='contig_id',
        ),
        migrations.RemoveField(
            model_name='contig',
            name='coverage',
        ),
        migrations.RemoveField(
            model_name='contig',
            name='fpkm',
        ),
        migrations.RemoveField(
            model_name='contig',
            name='fpkm_bbmap',
        ),
        migrations.RemoveField(
            model_name='contig',
            name='frags_mapped',
        ),
        migrations.RemoveField(
            model_name='contig',
            name='reads_mapped',
        ),
        migrations.RemoveField(
            model_name='contig',
            name='rpkm_bbmap',
        ),
        migrations.RemoveField(
            model_name='contig',
            name='taxa',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='bases',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='besthit',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='coverage',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='end',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='fasta_len',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='fasta_offset',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='fpkm',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='fpkm_bbmap',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='frags_mapped',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='gene_id',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='hits',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='lca',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='reads_mapped',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='rpkm',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='rpkm_bbmap',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='start',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='strand',
        ),
        migrations.RemoveField(
            model_name='gene',
            name='taxa',
        ),
        migrations.DeleteModel(
            name='Alignment',
        ),
    ]
