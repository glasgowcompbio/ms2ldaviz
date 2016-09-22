# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0038_auto_20160922_1212'),
    ]

    operations = [
        migrations.AlterField(
            model_name='intensityinstance',
            name='intensity',
            field=models.FloatField(null=True),
        ),
    ]
