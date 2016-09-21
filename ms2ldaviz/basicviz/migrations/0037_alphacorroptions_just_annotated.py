# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0036_alphacorroptions_max_edges'),
    ]

    operations = [
        migrations.AddField(
            model_name='alphacorroptions',
            name='just_annotated',
            field=models.BooleanField(default=False),
            preserve_default=False,
        ),
    ]
