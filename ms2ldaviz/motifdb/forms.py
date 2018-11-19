from django import forms
from django.contrib.auth.models import User
from django.db.models import Q

from motifdb.models import MDBMotifSet


class MatchMotifDBForm(forms.Form):
    motif_set = forms.ModelChoiceField(queryset = MDBMotifSet.objects.all(),required=True)
    min_score_to_save = forms.FloatField(required = True,initial=0.5)
