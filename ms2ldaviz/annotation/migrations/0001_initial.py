# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0048_userexperiment_permission'),
    ]

    operations = [
        migrations.CreateModel(
            name='SubstituentInstance',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('probability', models.FloatField()),
                ('motif', models.ForeignKey(to='basicviz.Mass2Motif', on_delete=models.CASCADE)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='SubstituentTerm',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(unique=True, max_length=128)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='TaxaInstance',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('probability', models.FloatField()),
                ('motif', models.ForeignKey(to='basicviz.Mass2Motif', on_delete=models.CASCADE)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='TaxaTerm',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(unique=True, max_length=128)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='taxainstance',
            name='taxterm',
            field=models.ForeignKey(to='annotation.TaxaTerm', on_delete=models.CASCADE),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='substituentinstance',
            name='subterm',
            field=models.ForeignKey(to='annotation.SubstituentTerm', on_delete=models.CASCADE),
            preserve_default=True,
        ),
    ]
