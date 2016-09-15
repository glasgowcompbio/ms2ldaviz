# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0027_multifileexperiment_pca'),
    ]

    operations = [
        migrations.AddField(
            model_name='multifileexperiment',
            name='alpha_matrix',
            field=models.TextField(null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='multifileexperiment',
            name='degree_matrix',
            field=models.TextField(null=True),
            preserve_default=True,
        ),
    ]
