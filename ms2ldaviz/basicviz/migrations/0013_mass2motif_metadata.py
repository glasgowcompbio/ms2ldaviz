# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0012_experiment_description'),
    ]

    operations = [
        migrations.AddField(
            model_name='mass2motif',
            name='metadata',
            field=models.CharField(max_length=1024, null=True),
            preserve_default=True,
        ),
    ]
