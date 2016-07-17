# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0008_documentmass2motif'),
    ]

    operations = [
        migrations.CreateModel(
            name='FeatureMass2MotifInstance',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('probability', models.FloatField()),
                ('feature', models.ForeignKey(to='basicviz.Feature')),
                ('mass2motif', models.ForeignKey(to='basicviz.Mass2Motif')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
