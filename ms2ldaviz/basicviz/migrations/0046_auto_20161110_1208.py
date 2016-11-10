# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0045_auto_20161004_2042'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mass2motif',
            name='metadata',
            field=models.CharField(max_length=1048576, null=True),
        ),
    ]
