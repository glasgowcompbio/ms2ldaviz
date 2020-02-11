# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0010_featuremass2motifinstance_document'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='featuremass2motifinstance',
            name='document',
        ),
        migrations.RemoveField(
            model_name='featuremass2motifinstance',
            name='feature',
        ),
        migrations.AddField(
            model_name='featuremass2motifinstance',
            name='featureinstance',
            field=models.ForeignKey(default=1, to='basicviz.FeatureInstance', on_delete=models.CASCADE),
            preserve_default=False,
        ),
    ]
