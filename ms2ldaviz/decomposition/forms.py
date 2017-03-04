# forms.py for decomposition
from django import forms

from decomposition.models import Decomposition

class DecompVizForm(forms.Form):
    min_degree = forms.IntegerField(required = True, initial = 10,label="Minimum topic degree for inclusion")

class NewDecompositionForm(forms.ModelForm):
	class Meta:
		model = Decomposition
		exclude = ['experiment','status']

		# todo: add something here so that only motifsets with the correct feature set can be seen
