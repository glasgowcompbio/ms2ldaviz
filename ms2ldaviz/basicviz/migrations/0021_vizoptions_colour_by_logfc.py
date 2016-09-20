# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0020_userexperiment'),
    ]

    operations = [
        migrations.AddField(
            model_name='vizoptions',
            name='colour_by_logfc',
            field=models.BooleanField(default=False),
            preserve_default=False,
        ),
    ]
