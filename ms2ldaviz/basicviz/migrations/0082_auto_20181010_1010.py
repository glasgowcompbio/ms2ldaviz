# -*- coding: utf-8 -*-
# Generated by Django 1.10.5 on 2018-10-10 10:10
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('basicviz', '0081_auto_20181009_1157'),
    ]

    operations = [
        migrations.CreateModel(
            name='Doc2Sub',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fragatoms', models.CharField(max_length=128)),
                ('document', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='basicviz.Document')),
            ],
        ),
        migrations.CreateModel(
            name='MagmaSub',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('smiles', models.TextField()),
                ('mol_string', models.TextField()),
            ],
        ),
        migrations.AddField(
            model_name='doc2sub',
            name='sub',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='basicviz.MagmaSub'),
        ),
    ]