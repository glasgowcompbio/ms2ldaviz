# -*- coding: utf-8 -*-
# Generated by Django 1.11.16 on 2020-06-17 21:36
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '090_longer_doc_name'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='name',
            field=models.CharField(max_length=1024),
        ),
        migrations.AlterField(
            model_name='experiment',
            name='experiment_ms2_format',
            field=models.CharField(choices=[(b'0', b'mzML'), (b'1', b'msp'), (b'2', b'mgf'), (b'3', b'upload'), (b'4', b'uploadgensim')], default=b'0', max_length=128, null=True),
        ),
    ]