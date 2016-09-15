# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0028_auto_20160913_1408'),
    ]

    operations = [
        migrations.AddField(
            model_name='vizoptions',
            name='random_seed',
            field=models.CharField(default='hello', max_length=128),
            preserve_default=False,
        ),
    ]
