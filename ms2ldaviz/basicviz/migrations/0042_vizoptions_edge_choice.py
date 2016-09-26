# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0041_documentmass2motif_overlap_score'),
    ]

    operations = [
        migrations.AddField(
            model_name='vizoptions',
            name='edge_choice',
            field=models.CharField(default='probability', max_length=128),
            preserve_default=False,
        ),
    ]
