import tempfile

from django.apps import apps
from django.conf import settings
from django.core.management import call_command
from django.test import Client, override_settings, TestCase
from django.urls import reverse

from mibios.umrad.model_utils import Model
from .models import Sample

from mibios.glamr import urls0


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
        for view_name in ['dbinfo', 'sample_status']:
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
class LoadMetaDataTests(TestCase):

    def setUp(self):
        self.tmpd = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpd.cleanup()

    def test_load_all_meta_data(self):
        if not settings.OMICS_DATA_ROOT.is_dir():
            self.skipTest(f'{settings.OMICS_DATA_ROOT=} does not exist')

        if not settings.GLAMR_META_ROOT.is_dir():
            self.skipTest(f'{settings.GLAMR_META_ROOT=} does not exist')

        with self.settings(IMPORT_DIFF_DIR=self.tmpd.name):
            Sample.loader.load_all_meta_data()
            call_command(
                'dumpdata',
                '--all',
                '--indent=4',
                f'--output={settings.TEST_FIXURES_DIR / "test_metadata.json"}',
                'glamr.sample',
                'glamr.reference',
                'glamr.dataset',
            )


class ViewTests(EmptyDBViewTests):
    @classmethod
    def setUpTestData(cls):
        call_command(
            'loaddata',
            f'{settings.TEST_FIXURES_DIR / "test_metadata.json"}',
        )
