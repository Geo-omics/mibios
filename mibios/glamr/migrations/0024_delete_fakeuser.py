# Generated by Django 3.2.19 on 2024-08-21 17:10

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('admin', '0003_logentry_add_action_flag_choices'),
        ('mibios', '0016_importfile_note'),
        ('glamr', '0023_auto_20240807_1714'),
    ]

    operations = [
        migrations.DeleteModel(
            name='FakeUser',
        ),
    ]