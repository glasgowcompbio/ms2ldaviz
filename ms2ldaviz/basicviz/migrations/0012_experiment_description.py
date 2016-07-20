# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0011_auto_20160717_2121'),
    ]

    operations = [
        migrations.AddField(
            model_name='experiment',
            name='description',
            field=models.CharField(max_length=1024, null=True),
            preserve_default=True,
        ),
    ]
