# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0037_alphacorroptions_just_annotated'),
    ]

    operations = [
        migrations.CreateModel(
            name='IntensityInstance',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('intensity', models.FloatField()),
                ('document', models.ForeignKey(to='basicviz.Document', null=True)),
                ('experiment', models.ForeignKey(to='basicviz.Experiment', null=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='PeakSet',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('mz', models.FloatField()),
                ('rt', models.FloatField()),
                ('multifileexperiment', models.ForeignKey(to='basicviz.MultiFileExperiment')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='intensityinstance',
            name='peakset',
            field=models.ForeignKey(to='basicviz.PeakSet'),
            preserve_default=True,
        ),
    ]
