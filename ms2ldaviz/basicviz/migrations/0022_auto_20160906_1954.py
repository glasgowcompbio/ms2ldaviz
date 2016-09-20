# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0021_vizoptions_colour_by_logfc'),
    ]

    operations = [
        migrations.AddField(
            model_name='vizoptions',
            name='discrete_colour',
            field=models.BooleanField(default=False),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='vizoptions',
            name='lower_colour_perc',
            field=models.IntegerField(default=10),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='vizoptions',
            name='upper_colour_perc',
            field=models.IntegerField(default=90),
            preserve_default=False,
        ),
    ]
