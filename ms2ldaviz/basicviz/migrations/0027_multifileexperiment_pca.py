# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0026_alpha_multifileexperiment_multilink'),
    ]

    operations = [
        migrations.AddField(
            model_name='multifileexperiment',
            name='pca',
            field=models.TextField(null=True),
            preserve_default=True,
        ),
    ]
