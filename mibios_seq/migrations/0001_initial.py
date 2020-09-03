# Generated by Django 2.2.14 on 2020-08-28 20:35

from django.db import migrations, models
import django.db.models.deletion
import mibios.models


class Migration(migrations.Migration):
    """
    Sequencing models transition

    The models were copy-pasted from the hhcd app and the tables named
    appropiately in migration 0008_sequencing_transition, so here nothing needs
    to be done to the database.
    """

    initial = True

    dependencies = [
        ('mibios', '0006_snapshot_jsondump'),
        ('hhcd', '0008_sequencing_transition'),
    ]

    state_ops = [
        migrations.CreateModel(
            name='ASV',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('number', models.PositiveIntegerField()),
                ('history', models.ManyToManyField(to='mibios.ChangeRecord')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Taxon',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('taxid', models.PositiveIntegerField()),
                ('organism', models.CharField(max_length=100)),
                ('history', models.ManyToManyField(to='mibios.ChangeRecord')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='Strain',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('asv', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='mibios_seq.ASV')),

                ('history', models.ManyToManyField(to='mibios.ChangeRecord')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.CreateModel(
            name='SequencingRun',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('serial', models.CharField(max_length=50)),
                ('number', models.PositiveSmallIntegerField()),
                ('path', models.CharField(blank=True, max_length=2000)),
                ('history', models.ManyToManyField(to='mibios.ChangeRecord')),
            ],
            options={
                'ordering': ['serial', 'number'],
                'unique_together': {('serial', 'number')},
            },
        ),
        migrations.CreateModel(
            name='Sequencing',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=100, unique=True)),
                ('control', models.CharField(blank=True, choices=[('mock', 'mock'), ('water', 'water'), ('blank', 'blank'), ('plate', 'plate'), ('other', 'other')], max_length=50)),
                ('r1_file', models.CharField(blank=True, max_length=300, null=True, unique=True)),
                ('r2_file', models.CharField(blank=True, max_length=300, null=True, unique=True)),
                ('plate', models.PositiveSmallIntegerField(blank=True, null=True)),
                ('plate_position', models.CharField(blank=True, max_length=10)),
                ('snumber', models.PositiveSmallIntegerField(blank=True, null=True)),
                ('history', models.ManyToManyField(to='mibios.ChangeRecord')),
                ('note', models.ManyToManyField(blank=True, to='hhcd.Note')),
                ('run', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='mibios_seq.SequencingRun')),
                ('sample', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='hhcd.FecalSample')),
            ],
            options={
                'ordering': ['name'],
                'unique_together': {('run', 'snumber'), ('run', 'plate', 'plate_position')},
            },
        ),
        migrations.CreateModel(
            name='Community',
            fields=[
                ('id', mibios.models.AutoField(primary_key=True, serialize=False)),
                ('asv', models.ManyToManyField(to='mibios_seq.ASV')),
                ('history', models.ManyToManyField(to='mibios.ChangeRecord')),
                ('seqs', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='mibios_seq.Sequencing')),
            ],
            options={
                'abstract': False,
            },
        ),
        migrations.AddField(
            model_name='asv',
            name='taxon',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='mibios_seq.Taxon'),
        ),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=state_ops,
            database_operations=[],  # tables exist already
        )
    ]