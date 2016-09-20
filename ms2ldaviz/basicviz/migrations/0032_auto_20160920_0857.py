# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0031_auto_20160920_0854'),
    ]

    operations = [
        migrations.RenameField(
            model_name='alphacorroptions',
            old_name='distance_type',
            new_name='distance_score',
        ),
    ]
