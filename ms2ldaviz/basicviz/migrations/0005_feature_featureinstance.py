# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0004_document_metadata'),
    ]

    operations = [
        migrations.CreateModel(
            name='Feature',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(max_length=32)),
                ('experiment', models.ForeignKey(to='basicviz.Experiment')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='FeatureInstance',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('intensity', models.FloatField()),
                ('doc', models.ForeignKey(to='basicviz.Document')),
                ('feature', models.ForeignKey(to='basicviz.Feature')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
