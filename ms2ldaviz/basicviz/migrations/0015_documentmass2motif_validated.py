# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0014_auto_20160721_1427'),
    ]

    operations = [
        migrations.AddField(
            model_name='documentmass2motif',
            name='validated',
            field=models.NullBooleanField(),
            preserve_default=True,
        ),
    ]
