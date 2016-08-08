# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0018_vizoptions'),
    ]

    operations = [
        migrations.AddField(
            model_name='vizoptions',
            name='just_annotated_docs',
            field=models.BooleanField(default=False),
            preserve_default=False,
        ),
    ]
