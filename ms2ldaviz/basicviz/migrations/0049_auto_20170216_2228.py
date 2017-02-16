# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations

def update_experiment_status(apps, schema_editor):
    Experiment = apps.get_model('basicviz', 'Experiment')
    for exp in Experiment.objects.all():
        # existing experiments have status 'all loaded', so we want to change them all to '1' (Ready)
        if exp.status != '0':
            exp.status = '1'
        exp.save()

class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0048_userexperiment_permission'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='status',
            field=models.CharField(default=b'1', max_length=128, null=True, choices=[(b'0', b'Pending'), (b'1', b'Ready')]),
        ),
        migrations.RunPython(update_experiment_status),
    ]
