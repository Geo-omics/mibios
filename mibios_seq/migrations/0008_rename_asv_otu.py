# Generated by Django 2.2.14 on 2020-10-08 14:17

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mibios', '0010_auto_20200923_1607'),
        ('mibios_seq', '0007_abundanceimportfile'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='ASV',
            new_name='OTU',
        ),
    ]