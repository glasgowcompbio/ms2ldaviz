# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0035_alphacorroptions_normalise_alphas'),
    ]

    operations = [
        migrations.AddField(
            model_name='alphacorroptions',
            name='max_edges',
            field=models.IntegerField(default=1000),
            preserve_default=False,
        ),
    ]
