# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0042_vizoptions_edge_choice'),
    ]

    operations = [
        migrations.CreateModel(
            name='SystemOptions',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('key', models.CharField(unique=True, max_length=124)),
                ('value', models.CharField(max_length=124)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
