# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0005_feature_featureinstance'),
    ]

    operations = [
        migrations.RenameField(
            model_name='featureinstance',
            old_name='doc',
            new_name='document',
        ),
    ]
