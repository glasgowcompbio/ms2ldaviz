# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0047_auto_20161123_1640'),
    ]

    operations = [
        migrations.AddField(
            model_name='userexperiment',
            name='permission',
            field=models.CharField(default='view', max_length=24),
            preserve_default=False,
        ),
    ]
