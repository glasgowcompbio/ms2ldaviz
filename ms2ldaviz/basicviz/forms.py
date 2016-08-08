from django import forms
from basicviz.models import Mass2Motif

class Mass2MotifMetadataForm(forms.Form):
	metadata = forms.CharField(max_length = 256,required = False,widget=forms.TextInput(attrs={'size':'80'}))

class DocFilterForm(forms.Form):
	annotated_only = forms.BooleanField(required = False,label='Restrict to annotated docs?')
	min_annotated_topics = forms.IntegerField(required = False,initial = 0,label='Minimum number of annotated topics per doc')
	topic_threshold = forms.DecimalField(required = False,initial = 0.0,label='Probability threshold for topic inclusion')
	only_show_annotated = forms.BooleanField(required = False,initial = False,label='Only display annotated topics')

class ValidationForm(forms.Form):
	p_thresh = forms.DecimalField(required = True,initial = 0.05,label='Enter document -> mass2motif membership threshold')