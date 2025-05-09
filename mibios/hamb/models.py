from django.db import models
from django.urls import reverse

from mibios.omics.models import AbstractDataset, AbstractSample
from mibios.umrad.model_utils import (
    opt, ch_opt, fk_opt, Model,
)

from .managers import HostManager, SampleManager


class Dataset(AbstractDataset):
    label = models.TextField(max_length=16, unique=True)

    def __str__(self):
        return self.label

    def get_absolute_url(self):
        return reverse('dataset_detail', kwargs=dict(pk=self.pk))


class Host(Model):
    label = models.TextField(max_length=16, unique=True)
    common_name = models.TextField()
    age_years = models.PositiveIntegerField(**opt)
    description = models.TextField(**ch_opt)
    health_state = models.TextField(**ch_opt)

    objects = HostManager()

    def __str__(self):
        return self.label

    def get_absolute_url(self):
        return reverse('host_detail', kwargs=dict(pk=self.pk))


class Sample(AbstractSample):
    label = models.TextField(max_length=16)
    biosample = models.TextField(max_length=16, **ch_opt)
    host = models.ForeignKey(Host, **fk_opt)
    source_material = models.TextField(max_length=32, blank=True)
    control = models.TextField(max_length=8, blank=True)

    objects = SampleManager()

    class Meta:
        constraints = (
            models.UniqueConstraint(
                fields=('label', 'dataset'),
                name='uniq_label_dataset',
            ),
        )

    def __str__(self):
        return self.label

    def get_absolute_url(self):
        return reverse('sample_detail', kwargs=dict(pk=self.pk))
