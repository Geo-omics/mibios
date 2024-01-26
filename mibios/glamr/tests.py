import tempfile
from unittest import skipUnless

from django.apps import apps
from django.conf import settings
from django.core.management import call_command
from django.test import Client, override_settings, TestCase
from django.urls import reverse

from mibios.umrad.model_utils import Model
from .models import Sample


@override_settings(ROOT_URLCONF='mibios.glamr.urls0')
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

    @skipUnless(settings.INTERNAL_DEPLOYMENT is True, 'INTERNAL_DEPLOYMENT not enabled')  # noqa:E501
    def test_internal_pages(self):
        for view_name in ['dbinfo', 'sample_status']:
            with self.subTest(view_name=view_name):
                resp = self.client.get(reverse(view_name))
                self.assertEqual(resp.status_code, 200)

    @skipUnless(settings.ENABLE_TEST_URLS is True, 'test urls not enabled')
    def test_minitest_pages(self):
        for url in ['/minitest/', '/basetest/']:
            with self.subTest(url=url):
                resp = self.client.get(url)
                self.assertEqual(resp.status_code, 200)

        url = '/errortest/'
        with self.subTest(url=url):
            c = Client()
            c.raise_request_exception = False
            resp = c.get(url)
            self.assertEqual(resp.status_code, 500)


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
