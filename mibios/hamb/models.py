from django.db import models

from mibios.omics.models import AbstractDataset, AbstractSample
from mibios.umrad.model_utils import (
    digits, opt, ch_opt, fk_req, fk_opt, uniq_opt, Model,
)


class Dataset(AbstractDataset):
    label = models.TextField(max_length=16, unique=True)


class Host(Model):
    label = models.TextField(max_length=16, unique=True)
    common_name = models.TextField()
    age_years = models.PositiveIntegerField(**opt)
    description = models.TextField(**ch_opt)
    health_state = models.TextField(**ch_opt)


class Sample(AbstractSample):
    label = models.TextField(max_length=16)
    biosample = models.TextField(max_length=16, **ch_opt)
    host = models.ForeignKey(Host, **fk_opt)
    source_material = models.TextField(max_length=32)
    control = models.TextField(max_length=8, blank=True)

    class Meta:
        constraints = (
            models.UniqueConstraint(
                fields=('label', 'dataset'),
                name='uniq_label_dataset',
            ),
        )
