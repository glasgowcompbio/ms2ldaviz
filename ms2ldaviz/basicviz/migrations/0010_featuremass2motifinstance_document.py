# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0009_featuremass2motifinstance'),
    ]

    operations = [
        migrations.AddField(
            model_name='featuremass2motifinstance',
            name='document',
            field=models.ForeignKey(default=1, to='basicviz.Document'),
            preserve_default=False,
        ),
    ]
