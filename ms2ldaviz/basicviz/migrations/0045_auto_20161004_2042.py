# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0044_systemoptions_experiment'),
    ]

    operations = [
        migrations.AlterField(
            model_name='systemoptions',
            name='key',
            field=models.CharField(max_length=124),
        ),
        migrations.AlterUniqueTogether(
            name='systemoptions',
            unique_together=set([('key', 'experiment')]),
        ),
    ]
