# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0025_multifileexperiment_multilink'),
    ]

    operations = [
        migrations.CreateModel(
            name='Alpha',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('value', models.FloatField()),
                ('mass2motif', models.ForeignKey(to='basicviz.Mass2Motif')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
