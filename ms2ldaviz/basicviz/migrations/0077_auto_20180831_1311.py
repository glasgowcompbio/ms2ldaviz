# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2018-08-31 13:11
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0076_auto_20180831_1310'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='csv_rt_units',
            field=models.CharField(blank=True, choices=[(b'minutes', b'minutes'), (b'seconds', b'seconds')], default=b'seconds', max_length=128, null=True),
        ),
    ]