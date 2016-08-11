# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0015_documentmass2motif_validated'),
    ]

    operations = [
        migrations.AddField(
            model_name='experiment',
            name='status',
            field=models.CharField(max_length=128, null=True),
            preserve_default=True,
        ),
        migrations.AlterField(
            model_name='document',
            name='metadata',
            field=models.CharField(max_length=2048, null=True),
        ),
    ]
