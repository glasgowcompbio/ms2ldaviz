# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0013_mass2motif_metadata'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='name',
            field=models.CharField(max_length=32),
        ),
    ]
