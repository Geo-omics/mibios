# Generated by Django 2.2 on 2020-07-13 15:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mibios', '0002_auto_20200713_1115'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fecalsample',
            name='note',
            field=models.ManyToManyField(blank=True, to='mibios.Note'),
        ),
        migrations.AlterField(
            model_name='note',
            name='text',
            field=models.TextField(blank=True, max_length=5000),
        ),
        migrations.AlterField(
            model_name='participant',
            name='has_consented',
            field=models.BooleanField(default=False, help_text='Corresponds to the Use_Data field in several original tables'),
        ),
        migrations.AlterField(
            model_name='participant',
            name='has_consented_future',
            field=models.BooleanField(blank=True, help_text='Use Data in Unspecified Future Research', null=True),
        ),
        migrations.AlterField(
            model_name='participant',
            name='note',
            field=models.ManyToManyField(blank=True, to='mibios.Note'),
        ),
        migrations.AlterField(
            model_name='sequencing',
            name='note',
            field=models.ManyToManyField(blank=True, to='mibios.Note'),
        ),
    ]
