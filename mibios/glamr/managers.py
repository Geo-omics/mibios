from django.apps import apps
from django.core import checks
from django.db.models import Manager as DjangoManager

from mibios.umrad.manager import Manager


class dbstatManager(DjangoManager):
    def get_queryset(self):
        # see: https://www.sqlite.org/dbstat.html
        # use extra+where instead of filter() since aggregate is not a field
        return super().get_queryset().extra(where=['aggregate = True'])


def check_dataset_access(app_configs, **kwargs):
    """
    System check for Dataset.access

    check that access field is synchronized with restricted_to
    """
    Dataset = apps.get_model('glamr.Dataset')
    qs = Dataset.objects.only('access').prefetch_related('restricted_to')
    good = []
    bad = []

    try:
        qs_list = list(qs)
    except Exception as e:
        return [checks.Warning(
            f'Unable to check Dataset.access consistency: {e}',
            hint='apply all migrations, ensure you have a DB connection',
            id='glamr.W011',
        )]

    for obj in qs_list:
        groupids = sorted((i.pk for i in obj.restricted_to.all())) or [0]
        if groupids == obj.access:
            good.append(obj)
        else:
            bad.append(obj)

    errors = []
    if bad:
        if good:
            errors.append(checks.Warning(
                f'Found inconsistent dataset.access content for {len(bad)} '
                f'datasets while {len(good)} datasets were good.',
                hint='Run Dataset.objects.update_access()',
                id='glamr.W007',
            ))
        else:
            errors.append(checks.Warning(
                f'All dataset.access content is inconsistent (for {len(bad)} '
                f'datasets)',
                hint='Run Dataset.objects.update_access()',
                id='glamr.W008',
            ))
    return errors


def check_sample_access(app_configs, **kwargs):
    """
    System check for SeqSample.access and Sample.access

    check that access field is synchronized with Dataset.restricted_to
    """
    SeqSample = apps.get_model('omics', 'SeqSample')
    qs = SeqSample.objects.only(
        'access', 'parent__access', 'parent__dataset__dataset_id',
    )
    qs = qs.prefetch_related('parent__dataset__restricted_to')
    good = []
    bad = []

    try:
        qs_list = list(qs)
    except Exception as e:
        return [checks.Warning(
            f'Unable to check access fields consistency: {e}',
            hint='apply all migrations, ensure you have a DB connection',
            id='glamr.W012',
        )]

    for obj in qs_list:
        groupids = sorted(i.pk for i in obj.parent.dataset.restricted_to.all())
        if not groupids:
            groupids = [0]
        if groupids == obj.access == obj.parent.access:
            good.append(obj)
        else:
            bad.append(obj)

    errors = []
    if bad:
        if good:
            errors.append(checks.Warning(
                f'Found inconsistent Sample.access content for {len(bad)} '
                f'samples while {len(good)} samples were good.',
                hint='Run Sample.objects.update_access()',
                id='glamr.W009',
            ))
        else:
            errors.append(checks.Warning(
                f'All Sample.access content is inconsistent (for {len(bad)} '
                f'samples)',
                hint='Run Sample.objects.update_access()',
                id='glamr.W010',
            ))
    return errors


class DatasetManager(Manager):
    def restriction_changed(self, **kwargs):
        """
        Update access fields when restricted_to relation changes

        Receiver for signals when restricted_to m2m relation is changed
        """
        # NOTE: it's not clear to me why this signal receiver works as regular
        # method, how is self populated?
        instance = kwargs.get('instance')
        action = kwargs.get('action')

        if action not in ('post_add', 'post_remove', 'post_clear'):
            return

        if kwargs.get('reverse'):
            raise RuntimeError('not implemented, what to do here?')

        if isinstance(instance, self.model):
            self.all().filter(pk=instance.pk).update_access()
        else:
            raise RuntimeError('what to do here?')
