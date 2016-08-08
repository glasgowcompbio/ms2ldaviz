# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0017_auto_20160808_1939'),
    ]

    operations = [
        migrations.CreateModel(
            name='VizOptions',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('edge_thresh', models.FloatField()),
                ('min_degree', models.IntegerField()),
                ('experiment', models.ForeignKey(to='basicviz.Experiment')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
