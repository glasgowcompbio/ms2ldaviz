# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0030_merge'),
    ]

    operations = [
        migrations.CreateModel(
            name='AlphaCorrOptions',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('edge_thresh', models.FloatField()),
                ('distance_type', models.CharField(max_length=24)),
                ('multifileexperiment', models.ForeignKey(to='basicviz.MultiFileExperiment', on_delete=models.CASCADE)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='multifileexperiment',
            name='alpha_matrix',
            field=models.TextField(null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='multifileexperiment',
            name='degree_matrix',
            field=models.TextField(null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='multifileexperiment',
            name='pca',
            field=models.TextField(null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='vizoptions',
            name='random_seed',
            field=models.CharField(default='hello', max_length=128),
            preserve_default=False,
        ),
    ]
