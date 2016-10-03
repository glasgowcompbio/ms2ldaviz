# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0043_systemoptions'),
    ]

    operations = [
        migrations.AddField(
            model_name='systemoptions',
            name='experiment',
            field=models.ForeignKey(to='basicviz.Experiment', null=True),
            preserve_default=True,
        ),
    ]
