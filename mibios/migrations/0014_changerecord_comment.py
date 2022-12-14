# Generated by Django 2.2.14 on 2020-12-09 17:10
# plus manual change

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mibios', '0013_auto_20201103_1423'),
    ]

    operations = [
        migrations.RenameField(
            model_name='changerecord',
            old_name='command_line',
            new_name='comment',
        ),
        migrations.AlterField(
            model_name='changerecord',
            name='comment',
            field=models.CharField(blank=True, help_text='Additional info, comment, or management command for import', max_length=200),
        ),
    ]
