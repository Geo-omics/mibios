"""
Tests for GLAMR

How to run tests with postgres:  As postgres user, create a test DB
"test_glamr" owned by the developer's role.  Then run tests with --keepdb
option.

How to get test  coverage:
python3 -m coverage run --branch --source=./mibios ./manage.py test
python3 -m coverage html -d cov_html

To create or re-create the test fixture run the AAALoadMetaDataTests test
first.

Whole site testing suggestion:
linkchecker [--user=<user>] -r <2|3> -F text --no-robots <url>
"""
import logging
import re
import tempfile

from django.apps import apps
from django.conf import settings
from django.contrib.auth.models import User
from django.core.management import call_command
from django.db import connections
from django.test import Client, override_settings, runner, TestCase, tag
from django.urls import reverse

from mibios.umrad.model_utils import Model
from .models import Dataset, Sample

from . import urls0, models as glamr_models
from .accounts import create_staff_group
from .views import GenericModelMixin, SearchMixin


class DiscoverRunner(runner.DiscoverRunner):
    """
    Test runner that excludes some tests by default.

    To run the excluded tests do e.g.:

        ./manage.py test --tag longrun
    """
    default_exclude_tags = ('longrun', )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # add default excludes to excludes, except those that are in tags
        self.exclude_tags.update(
            set(self.default_exclude_tags).difference(self.tags)
        )

    def run_tests(self, *args, **kwargs):
        logging.disable(logging.CRITICAL)
        try:
            return super().run_tests(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)


class TestUserMixin:
    TEST_USERS = {
        'Alice': 'aedf4wagb53@#FSDGs',
        'Bob': 'vcvewr4@$%$TFAdadsga',
    }

    @classmethod
    def setUpTestData(cls):
        User.objects.create_user('Alice', password=cls.TEST_USERS['Alice'],
                                 is_staff=True)
        User.objects.create_user('Bob', password=cls.TEST_USERS['Bob'])
        # admin pages check permissions, so the staff group has to be set up
        staff_grp = create_staff_group()
        staff_grp.user_set.add(*User.objects.filter(is_staff=True))

    def login(self, user):
        """ convenience wrapper around client.login() """
        if isinstance(user, User):
            username = user.username
        else:
            username = user
        return self.client.login(
            username=username,
            password=self.TEST_USERS[username],
        )


class TestDataMixin(TestUserMixin):
    """ Mixin for TestCase with a populated test database """
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        call_command(
            'loaddata',
            f'{settings.TEST_FIXTURES_DIR / "test_metadata.json"}',
        )


@override_settings(ROOT_URLCONF=urls0)
class EmptyDBViewTests(TestCase):
    """ test views that can run on an empty database """
    def test_frontpage(self):
        resp = self.client.get(reverse('frontpage'))
        self.assertEqual(resp.status_code, 200)

    def test_about_page(self):
        resp = self.client.get(reverse('about'))
        self.assertEqual(resp.status_code, 200)

    def test_about_history_page(self):
        resp = self.client.get(reverse('about-history'))
        self.assertEqual(resp.status_code, 200)

    def test_contact_page(self):
        resp = self.client.get(reverse('contact'))
        self.assertEqual(resp.status_code, 200)

    def test_data_generic_table_page(self):
        allowed = GenericModelMixin.get_allowed_models()
        for appname, app_models in apps.all_models.items():
            for model_name, model in app_models.items():
                if appname == 'omics' and model_name in ['sample', 'dataset']:
                    # those swapped out models mess up the assertEq logic below
                    continue
                if issubclass(model, Model):
                    with self.subTest(model_name=model_name):
                        kw = dict(model=model_name)
                        url = reverse('generic_table', kwargs=kw)
                        resp = self.client.get(url)
                        self.assertEqual(
                            resp.status_code,
                            200 if model in allowed else 404
                        )

    def test_data_generic_table_404_page(self):
        kw = dict(model='nosuchmodel')
        resp = self.client.get(reverse('generic_table', kwargs=kw))
        self.assertEqual(resp.status_code, 404)

    def test_nonexisting_page(self):
        resp = self.client.get('/doesnotexist/')
        self.assertEqual(resp.status_code, 404)

    def test_test_view_pages(self):
        for url in ['/minitest/', '/basetest/']:
            with self.settings(ENABLE_TEST_VIEWS=True):
                with self.subTest(url=url):
                    resp = self.client.get(url)
                    self.assertEqual(resp.status_code, 200)
            with self.settings(ENABLE_TEST_VIEWS=False):
                with self.subTest(url=url):
                    resp = self.client.get(url)
                    self.assertEqual(resp.status_code, 404)

        url = '/errortest/'
        with self.settings(ENABLE_TEST_VIEWS=True):
            with self.subTest(url=url, view_enabled=True):
                c = Client()
                c.raise_request_exception = False
                resp = c.get(url)
                self.assertEqual(resp.status_code, 500)

        with self.settings(ENABLE_TEST_VIEWS=False):
            with self.subTest(url=url, view_enabled=False):
                c = Client()
                c.raise_request_exception = False
                resp = c.get(url)
                self.assertEqual(resp.status_code, 404)


@override_settings(ROOT_URLCONF=urls0)
@override_settings(
    AUTHENTICATION_BACKENDS=['django.contrib.auth.backends.ModelBackend']
)
class StaffAccessTests(TestDataMixin, TestCase):
    """
    Test accessing certain pages while logged in as staff or non-staff
    """
    def _get_restricted_page(self, view):
        """
        request a staff-only page as different users

        The deny response status code depends a bit on the page.
        """
        # admin views redirect to the login page, even if user is logged in
        # already, on glamr's staff pages we'll just give out a 403
        if view.startswith('glamr_admin:'):
            deny_status = 302
        else:
            deny_status = 403

        qs = User.objects.all()
        # need staff and non-staff for this test
        self.assertTrue(qs.filter(is_staff=True).exists())
        self.assertTrue(qs.filter(is_staff=False).exists())
        for user in qs:
            params = dict(view=view, user=user, is_staff=user.is_staff)
            with self.subTest(**params):
                self.assertTrue(self.login(user))
                resp = self.client.get(reverse(view))
                self.assertEqual(
                    resp.status_code,
                    200 if user.is_staff else deny_status
                )

        with self.subTest(view=view, user='anonymous'):
            self.client.logout()
            resp = self.client.get(reverse(view))
            self.assertEqual(resp.status_code, deny_status)

    def test_simple_login(self):
        """ check that users were created correctly and that login() works """
        for i in User.objects.all():
            with self.subTest(user=i.username):
                self.assertTrue(self.login(i))

    def test_internal_pages(self):
        for view in ['dbinfo', 'sample_tracking']:
            self._get_restricted_page(view)

    def test_admin_pages(self):
        PREF = 'glamr_admin:'
        views = ['glamr_aboutinfo_changelist', 'glamr_credit_changelist',
                 'index']
        for view in views:
            self._get_restricted_page(PREF + view)

    def test_private_data_access(self):
        qs = Dataset.loader.filter(private=True)
        if not qs.exists():
            self.skipTest('no private datasets in test data')

        staff = User.objects.filter(is_staff=True).first()
        normal = User.objects.filter(is_staff=False).first()

        setno = qs.first().get_set_no()
        views = [
            ('dataset_sample_list', (setno, )),
            ('dataset', (setno, )),
        ]
        for user in [staff, normal]:
            self.assertTrue(self.login(user))
            for view_name, args in views:
                params = dict(user=user, is_staff=user.is_staff,
                              view=(view_name, args))
                with self.subTest(**params):
                    resp = self.client.get(reverse(view_name, args=args))
                    self.assertEqual(
                        resp.status_code,
                        200 if user.is_staff else 404
                    )


@override_settings(ROOT_URLCONF=urls0)
class AAALoadMetaDataTests(TestCase):
    """
    Test loading the meta data and save as a fixture

    The test class' name starts with AAA so it runs first and provides the
    fistures to be used by other tests.
    """

    def setUp(self):
        self.tmpd = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpd.cleanup()

    def test_load_all_meta_data(self):
        if not settings.OMICS_DATA_ROOT.is_dir():
            self.skipTest(f'{settings.OMICS_DATA_ROOT=} does not exist')

        if not settings.GLAMR_META_ROOT.is_dir():
            self.skipTest(f'{settings.GLAMR_META_ROOT=} does not exist')

        with self.settings(IMPORT_LOG_DIR=self.tmpd.name):
            Sample.loader.load_all_meta_data()
            call_command(
                'dumpdata',
                '--all',
                '--indent=4',
                '--natural-foreign',
                f'--output='
                f'{settings.TEST_FIXTURES_DIR / "test_metadata.json"}',
                'glamr.sample',
                'glamr.reference',
                'glamr.dataset',
                'auth.group',
            )

    def test_compile_extra_field_attributes(self):
        if not settings.GLAMR_META_ROOT.is_dir():
            self.skipTest(f'{settings.GLAMR_META_ROOT=} does not exist')

        call_command(
            'compile_extra_field_attributes',
            output_file=f'{self.tmpd.name}/extra_field_attrs_test.py',
        )


class LoaderTests(TestDataMixin, TestCase):
    def setUp(self):
        self.tmpd = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpd.cleanup()

    def test_loader_delete(self):
        """ test loader delete mode """
        if not settings.GLAMR_META_ROOT.is_dir():
            self.skipTest(f'{settings.GLAMR_META_ROOT=} does not exist')

        qs = glamr_models.Sample.objects.all()
        count = qs.count()

        # create a sample which does not exist in input data
        obj = qs.first()
        obj.sample_id = obj.sample_id + '_foo'
        obj.id = None  # trigger an INSERT
        obj.save()

        self.assertEqual(qs.count(), count + 1)

        with self.settings(IMPORT_LOG_DIR=self.tmpd.name):
            # should delete the above sample
            Sample.loader.load(delete=True, no_input=True)

        self.assertEqual(qs.count(), count)

        with self.assertRaises(glamr_models.Sample.DoesNotExist):
            qs.get(pk=obj.id)


class SearchTests(TestDataMixin, EmptyDBViewTests):
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        glamr_models.Searchable.objects.reindex()
        if connections['default'].vendor == 'postgresql':
            glamr_models.UniqueWord.objects.reindex()

    def test_search(self):
        search = glamr_models.Searchable.objects.search
        params = [
            dict(),
            dict(abundance=True),
            dict(models=['sample', 'dataset']),
            dict(models=['sample'], fields=['geo_loc_name']),
            dict(lookup='icontains'),
        ]
        for kwargs in params:
            with self.subTest(kwargs=kwargs):
                search('Lake Erie', **kwargs)

    def test_search_view(self):
        ANY = SearchMixin.ANY_MODEL_URL_PART
        for model in [ANY, 'sample', 'dataset']:
            for search_term in ['lake erie', 'lake nosuchthingneverfindthis']:
                with self.subTest(model=model, search_term=search_term):
                    urlkw = dict(model=model)
                    url = reverse('search_result', kwargs=urlkw)
                    qstr_data = dict(model=model, query=search_term)
                    r = self.client.get(url, data=qstr_data)
                    self.assertEqual(r.status_code, 200)


class AboutInfoTest(TestDataMixin, TestCase):
    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()

        c1 = glamr_models.Credit(
            name='foo', version='1.0', group=glamr_models.Credit.TOOL
        )
        c1.full_clean()
        c1.save()
        c2 = glamr_models.Credit(
            name='bar', version='3.2', group=glamr_models.Credit.DATA
        )
        c2.full_clean()
        c2.save()

        info = glamr_models.AboutInfo.objects.new()
        info.credits.add(c1, c2)
        glamr_models.AboutInfo.objects.publish()
        glamr_models.AboutInfo.objects.new()
        glamr_models.AboutInfo.objects.auto_update()

    def test_about_page(self):
        resp = self.client.get(reverse('about'))
        self.assertEqual(resp.status_code, 200)

    def test_about_history_page(self):
        resp = self.client.get(reverse('about-history'))
        self.assertEqual(resp.status_code, 200)


class DatasetTest(TestDataMixin, TestCase):
    def test_summaries(self):
        glamr_models.Dataset.objects.summary()
        glamr_models.Dataset.objects.summary(otherize=False)
        glamr_models.Dataset.objects.summary(otherize=False,
                                             as_dataframe=False)
        with self.assertRaises(RuntimeError):
            glamr_models.Dataset.objects.summary(as_dataframe=False)


class SampleTest(TestDataMixin, TestCase):
    def test_summaries(self):
        glamr_models.Sample.objects.summary()
        glamr_models.Sample.objects.summary(blank_sample_type=True)
        glamr_models.Sample.objects.summary(otherize=False)


class DeepLinkTests(TestDataMixin, TestCase):
    MAX_DEPTH = 2

    href_pat = re.compile(r'<a href="([/?][^"]+)"')
    """ pattern for local links """

    def do_test_for_url(self, url, depth, parent):
        if url in self.urls_tested:
            return

        if depth > self.MAX_DEPTH:
            self.urls_too_deep.add(url)
            return

        self.urls_tested.add(url)
        success = False
        with self.subTest(url=url, parent=parent):
            r = self.client.get(url)
            self.assertEqual(r.status_code, 200)
            success = True

        if not success:
            return

        if b'<!doctype html>' not in r.content[:50]:
            return

        for next_url in self.href_pat.findall(r.content.decode()):
            if next_url.startswith('?'):
                # a querystring-only URL, relative to parent
                # need to prepend parent's path to querystring
                path, _, _ = url.partition('?')
                next_url = path + next_url
            self.do_test_for_url(next_url, depth + 1, parent=url)

    def test_from_frontpage(self):
        self.urls_tested = set()
        self.urls_too_deep = set()
        self.do_test_for_url(reverse('frontpage'), depth=1, parent=None)
        print(f'\nnumber of URLs tested: {len(self.urls_tested)}')
        print(f'not tested but at next depth: {len(self.urls_too_deep)}')


@tag('longrun')
class VeryDeepLinkTests(DeepLinkTests):
    MAX_DEPTH = 3


class QuerySetIterateTests(TestDataMixin, TestCase):
    def test_model_iterate(self):
        qs = Sample.objects.all()
        kw_picks = [
            dict(chunk_size=1000),
            dict(cache=True),
            dict(cache=False),
        ]
        for kwargs in kw_picks:
            with self.subTest(iterate_kwargs=kwargs):
                list(qs.iterate(**kwargs))

    def test_values_list_iterate(self):
        qss = {
            'all_cols': Sample.objects.values_list(),
            'with_pk': Sample.objects.values_list('pk', 'sample_id'),
            'no_pk': Sample.objects.values_list('sample_id', 'dataset'),
        }
        kw_picks = [
            dict(chunk_size=1000),
            dict(cache=True),
            dict(cache=False),
        ]
        for qs in qss.keys():
            for kwargs in kw_picks:
                with self.subTest(iterate_kwargs=kwargs, queryset=qs):
                    list(qss[qs].iterate(**kwargs))
