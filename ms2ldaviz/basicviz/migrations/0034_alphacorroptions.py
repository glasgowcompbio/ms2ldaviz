# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0033_auto_20160920_0859'),
    ]

    operations = [
        migrations.CreateModel(
            name='AlphaCorrOptions',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('edge_thresh', models.FloatField()),
                ('distance_score', models.CharField(max_length=24)),
                ('multifileexperiment', models.ForeignKey(to='basicviz.MultiFileExperiment')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
