# Generated by Django 2.2.14 on 2020-11-12 15:47

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('mibios_seq', '0016_abundance_relative'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sequencing',
            name='sample',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='mibios_seq.Sample'),
        ),
    ]