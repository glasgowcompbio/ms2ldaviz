# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0024_extrausers'),
    ]

    operations = [
        migrations.AddField(
            model_name='vizoptions',
            name='colour_topic_by_score',
            field=models.BooleanField(default=False),
            preserve_default=False,
        ),
    ]
