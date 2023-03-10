from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('glamr', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='searchterm',
            name='field',
            field=models.CharField(db_index=True, default='foo', max_length=100),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='searchterm',
            name='has_hit',
            field=models.BooleanField(db_index=True, default=False),
        ),
    ]
