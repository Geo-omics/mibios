from django.apps import apps
from django.test import TestCase
from django.urls import reverse

from mibios.umrad.model_utils import Model


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
                    kw = dict(model=model_name)
                    resp = self.client.get(reverse('generic_table', kwargs=kw))
                    self.assertEqual(resp.status_code, 200)
