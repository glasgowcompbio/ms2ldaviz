# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0015_documentmass2motif_validated'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='metadata',
            field=models.CharField(max_length=2048, null=True),
        ),
    ]
