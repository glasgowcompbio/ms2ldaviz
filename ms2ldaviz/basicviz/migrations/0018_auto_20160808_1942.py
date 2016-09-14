# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0017_merge'),
    ]

    operations = [
        migrations.AlterField(
            model_name='feature',
            name='name',
            field=models.CharField(max_length=64),
        ),
    ]
