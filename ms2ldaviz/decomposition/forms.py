# forms.py for decomposition
from django import forms

class DecompVizForm(forms.Form):
    min_degree = forms.IntegerField(required = True, initial = 10,label="Minimum topic degree for inclusion")
