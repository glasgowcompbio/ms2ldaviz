# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0034_alphacorroptions'),
    ]

    operations = [
        migrations.AddField(
            model_name='alphacorroptions',
            name='normalise_alphas',
            field=models.BooleanField(default=True),
            preserve_default=False,
        ),
    ]
