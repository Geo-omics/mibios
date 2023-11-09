# Generated by Django 3.2.20 on 2023-11-09 22:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('glamr', '0009_aboutinfo'),
    ]

    operations = [
        migrations.CreateModel(
            name='dbstat',
            fields=[
                ('name', models.TextField(primary_key=True, serialize=False)),
                ('num_pages', models.IntegerField(db_column='pageno')),
                ('num_rows', models.IntegerField(db_column='ncell')),
            ],
            options={
                'db_table': 'dbstat',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='pg_class',
            fields=[
                ('name', models.TextField(db_column='relname', primary_key=True, serialize=False)),
                ('kind', models.CharField(choices=[('r', 'ordinary table'), ('i', 'index'), ('S', 'sequence'), ('t', 'TOAST table'), ('v', 'view'), ('m', 'materialized view'), ('c', 'composite type'), ('f', 'foreign table'), ('p', 'partitioned table'), ('I', 'partitioned index')], db_column='relkind', max_length=1)),
                ('num_pages', models.IntegerField(db_column='relpages')),
                ('num_rows', models.IntegerField(db_column='reltuples')),
            ],
            options={
                'db_table': 'pg_class',
                'managed': False,
            },
        ),
    ]
