"""
Tests for GLAMR

How to run tests with postgres:  As postgres user, create a test DB
"test_glamr" owned by the developer's role.  Then run tests with --keepdb
option.

How to get test  coverage:
python3 -m coverage run --branch --source=./mibios ./manage.py test
python3 -m coverage html -d cov_html
"""
import re
import tempfile

from django.apps import apps
from django.conf import settings
from django.core.management import call_command
from django.db import connections
from django.test import Client, override_settings, runner, TestCase, tag
from django.urls import reverse

from mibios.umrad.model_utils import Model
from .models import Sample

from . import urls0, models as glamr_models
from .views import SearchMixin


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


class TestDataMixin:
    """ Mixin for TestCase with a populated test database """
    @classmethod
    def setUpTestData(cls):
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
        for _, app_models in apps.all_models.items():
            for model_name, model in app_models.items():
                if issubclass(model, Model):
                    with self.subTest(model_name=model_name):
                        kw = dict(model=model_name)
                        url = reverse('generic_table', kwargs=kw)
                        resp = self.client.get(url)
                        self.assertEqual(resp.status_code, 200)

    def test_data_generic_table_404_page(self):
        kw = dict(model='nosuchmodel')
        resp = self.client.get(reverse('generic_table', kwargs=kw))
        self.assertEqual(resp.status_code, 404)

    def test_nonexisting_page(self):
        resp = self.client.get('/doesnotexist/')
        self.assertEqual(resp.status_code, 404)

    def test_internal_pages(self):
        for view_name in ['dbinfo', 'sample_tracking']:
            with self.settings(INTERNAL_DEPLOYMENT=True):
                with self.subTest(view_name=view_name, view_enabled=True):
                    resp = self.client.get(reverse(view_name))
                    self.assertEqual(resp.status_code, 200)
            with self.settings(INTERNAL_DEPLOYMENT=False):
                with self.subTest(view_name=view_name, view_enabled=False):
                    resp = self.client.get(reverse(view_name))
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

    def test_admin_pages(self):
        urls = ['/admin/', '/admin/glamr/', '/admin/glamr/aboutinfo/',
                '/admin/glamr/credit/']

        # The admin interface is setup during django's setup at module loading
        # time, so we can't dynamically enable/disable it via settings here.
        if not settings.INTERNAL_DEPLOYMENT:
            self.skipTest('INTERNAL_DEPLOYMENT is not True')
        if not settings.ENABLE_OPEN_ADMIN:
            self.skipTest('ENABLE_OPEN_ADMIN is not True')

        for i in urls:
            with self.subTest(url=i):
                self.assertEqual(self.client.get(i).status_code, 200)


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
                f'--output='
                f'{settings.TEST_FIXTURES_DIR / "test_metadata.json"}',
                'glamr.sample',
                'glamr.reference',
                'glamr.dataset',
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
