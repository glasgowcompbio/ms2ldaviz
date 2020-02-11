# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0007_mass2motif_mass2motifinstance'),
    ]

    operations = [
        migrations.CreateModel(
            name='DocumentMass2Motif',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('probability', models.FloatField()),
                ('document', models.ForeignKey(to='basicviz.Document', on_delete=models.CASCADE)),
                ('mass2motif', models.ForeignKey(to='basicviz.Mass2Motif', on_delete=models.CASCADE)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
