# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0024_vizoptions_colour_topic_by_score'),
    ]

    operations = [
        migrations.CreateModel(
            name='MultiFileExperiment',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(unique=True, max_length=128)),
                ('description', models.CharField(max_length=1024, null=True)),
                ('status', models.CharField(max_length=128, null=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='MultiLink',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('experiment', models.ForeignKey(to='basicviz.Experiment')),
                ('multifileexperiment', models.ForeignKey(to='basicviz.MultiFileExperiment')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
