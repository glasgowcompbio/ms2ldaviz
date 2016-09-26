# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0040_auto_20160922_1241'),
    ]

    operations = [
        migrations.AddField(
            model_name='documentmass2motif',
            name='overlap_score',
            field=models.FloatField(null=True),
            preserve_default=True,
        ),
    ]
