# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0046_auto_20161110_1208'),
    ]

    operations = [
        migrations.AddField(
            model_name='feature',
            name='max_mz',
            field=models.FloatField(null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='feature',
            name='min_mz',
            field=models.FloatField(null=True),
            preserve_default=True,
        ),
    ]
