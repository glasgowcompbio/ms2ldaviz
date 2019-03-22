# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin

from motifdb.models import MDBMotifSet,MDBMotif

# Register your models here.

admin.site.register(MDBMotifSet)
admin.site.register(MDBMotif)
