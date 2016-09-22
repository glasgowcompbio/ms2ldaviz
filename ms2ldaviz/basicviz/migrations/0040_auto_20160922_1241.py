# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0039_auto_20160922_1231'),
    ]

    operations = [
        migrations.AddField(
            model_name='peakset',
            name='original_file',
            field=models.CharField(max_length=124, null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='peakset',
            name='original_id',
            field=models.IntegerField(null=True),
            preserve_default=True,
        ),
    ]
