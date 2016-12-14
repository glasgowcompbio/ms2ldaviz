# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('annotation', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='substituentinstance',
            name='probability',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='taxainstance',
            name='probability',
            field=models.FloatField(null=True),
        ),
    ]
