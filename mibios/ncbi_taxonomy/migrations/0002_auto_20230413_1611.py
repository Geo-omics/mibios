# Generated by Django 3.2.18 on 2023-04-13 20:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ncbi_taxonomy', '0001_initial'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='taxname',
            options={'verbose_name': 'taxonomic name'},
        ),
        migrations.AddField(
            model_name='taxnode',
            name='ancestors',
            field=models.ManyToManyField(to='ncbi_taxonomy.TaxNode'),
        ),
    ]