# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0006_auto_20160717_2004'),
    ]

    operations = [
        migrations.CreateModel(
            name='Mass2Motif',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(max_length=32)),
                ('experiment', models.ForeignKey(to='basicviz.Experiment', on_delete=models.CASCADE)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Mass2MotifInstance',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('probability', models.FloatField()),
                ('feature', models.ForeignKey(to='basicviz.Feature', on_delete=models.CASCADE)),
                ('mass2motif', models.ForeignKey(to='basicviz.Mass2Motif', on_delete=models.CASCADE)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
