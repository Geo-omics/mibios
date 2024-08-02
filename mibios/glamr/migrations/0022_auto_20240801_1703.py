"""
Generated originally by Django 3.2.19 on 2024-08-01 21:03

Data migration added.  The data migrations imay make some assumptions that hold
for test or production data as of writing.  After the migration decimal values
may have trailing zeros remaining if the field migrated from DecimalField to
FreeDecimalField.  A Sample.loader.load_meta() should fix that.
"""
from functools import partial

from django.core.exceptions import ValidationError
from django.db import migrations, models
import mibios.glamr.fields


def fwd_before(apps, schema_editor, field_names=None, old_values=None):
    """
    Retrieve ond save riginal data and set all values to '0'

    The '0' can be converted to numeric types.
    """
    if field_names is None:
        raise ValueError('expecting a list of field names')

    if not isinstance(old_values, list) and len(old_values) != 0:
        raise ValueError('old_values: expecting an empty list')

    Sample = apps.get_model('glamr.Sample')
    samp_manager = Sample._meta.base_manager

    # saving values, replace them with 0
    old_values.extend(samp_manager.values('pk', *field_names))
    samp_manager.update(**{i: 0 for i in field_names})


def fwd_after(apps, schema_editor, old_values=None):
    """
    Restore old values, if possible

    Values that don't convert get set to NULL
    """
    if not isinstance(old_values, list):
        raise ValueError('old_values: expecting a list')

    Sample = apps.get_model('glamr.Sample')
    samp_manager = Sample._meta.base_manager

    # convert values (if possible)
    values = {}
    for vals in old_values:
        pk = vals.pop('pk')
        for k, v in vals.items():
            if v == '':
                vals[k] = None
            else:
                try:
                    v = Sample._meta.get_field(k).to_python(v)
                except ValidationError:
                    # NULL non-representable values
                    vals[k] = None
                else:
                    vals[k] = v

        values[pk] = vals

    # saving values (overwriting all those zeros from the before function)
    for obj in samp_manager.all():
        for k, v in values[obj.pk].items():
            setattr(obj, k, v)
        try:
            obj.full_clean()
        except ValidationError as e:
            print(f'{obj}: {e}')
        obj.save()


class Migration(migrations.Migration):

    dependencies = [
        ('glamr', '0021_auto_20240717_1623'),
    ]

    operations = [
        migrations.RunPython(fwd_before, migrations.RunPython.noop),
        migrations.AlterField(
            model_name='sample',
            name='ammonium',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=8, default=None, max_digits=10, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='atmospheric_temp',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=4, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='attenuation',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=1, default=None, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='bluegreen',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='calcium',
            field=models.PositiveIntegerField(blank=True, default=None, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='chlorophyl',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=7, default=None, max_digits=10, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='chlorophyl_fluoresence',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=5, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='col_dom',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=4, default=None, max_digits=6, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='conduc',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=5, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='crypto',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='cyano_sonde',
            field=models.PositiveIntegerField(blank=True, default=None, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='depth_location',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=5, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='depth_sediment',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='diatoms',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='diss_org_carb',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=4, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='diss_oxygen',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=5, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='diss_phos',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=4, default=None, max_digits=7, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='ext_anatox',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='ext_microcyst',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='ext_phyco',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=8, default=None, max_digits=11, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='green_algae',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='h2o2',
            field=models.PositiveIntegerField(blank=True, default=None, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='latitude',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=8, default=None, max_digits=10, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='longitude',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=8, default=None, max_digits=11, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='magnesium',
            field=models.PositiveIntegerField(blank=True, default=None, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='nitrate_nitrite',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=3, default=None, max_digits=6, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='nitrite',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=4, default=None, max_digits=9, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='orp',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=1, default=None, max_digits=4, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='par',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=6, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='part_microcyst',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=3, default=None, max_digits=5, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='part_org_carb',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=4, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='part_org_nitro',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='particulate_cyl',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=3, default=None, max_digits=4, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='phosphate',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=8, default=None, max_digits=10, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='phyco_fluoresence',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=4, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='potassium',
            field=models.PositiveIntegerField(blank=True, default=None, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='salinity',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='samp_vol_we_dna_ext',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=1, default=None, max_digits=5, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='secchi',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=4, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='silicate',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=3, default=None, max_digits=7, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='size_frac_low',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=5, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='size_frac_up',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=5, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='suspend_part_matter',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=5, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='suspend_vol_solid',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=4, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='temp',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=4, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='tot_microcyst_lcmsms',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=2, default=None, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='tot_nit',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=8, default=None, max_digits=11, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='total_sonde',
            field=models.PositiveIntegerField(blank=True, default=None, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='transmission',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=1, default=None, max_digits=3, null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='turbidity',
            field=mibios.glamr.fields.FreeDecimalField(blank=True, decimal_places=3, default=None, max_digits=6, null=True),
        ),
        migrations.RunPython(fwd_after, migrations.RunPython.noop),
    ]

    def __init__(self, *args, **kwargs):
        """
        Super special init to pass the field names to the RunPython forward
        function
        """
        super().__init__(*args, **kwargs)
        self.old_values = []
        run_python_before = self.operations[0]
        run_python_after = self.operations[-1]
        # get AlterField ops and extract field names
        field_names = [i.name for i in self.operations[1:-1]]

        # inject extra kwargs into functions
        run_python_before.code = partial(
            run_python_before.code,
            field_names=field_names,
            old_values=self.old_values,
        )
        run_python_after.code = partial(
            run_python_after.code,
            old_values=self.old_values,
        )
