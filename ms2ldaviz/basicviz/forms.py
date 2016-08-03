from django import forms
from basicviz.models import Mass2Motif

class Mass2MotifMetadataForm(forms.Form):
	metadata = forms.CharField(max_length = 256,widget=forms.TextInput(attrs={'size':'80'}))
