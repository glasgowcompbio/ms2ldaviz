from django import forms
from basicviz.models import Mass2Motif
from django.contrib.auth.models import User


class Mass2MotifMetadataForm(forms.Form):
	metadata = forms.CharField(max_length = 256,required = False,widget=forms.TextInput(attrs={'size':'80'}))

class DocFilterForm(forms.Form):
	annotated_only = forms.BooleanField(required = False,label='Restrict to annotated docs?')
	min_annotated_topics = forms.IntegerField(required = False,initial = 0,label='Minimum number of annotated topics per doc')
	topic_threshold = forms.DecimalField(required = False,initial = 0.0,label='Probability threshold for topic inclusion')
	only_show_annotated = forms.BooleanField(required = False,initial = False,label='Only display annotated topics')

class ValidationForm(forms.Form):
	p_thresh = forms.DecimalField(required = True,initial = 0.05,label='Enter document -> mass2motif membership threshold')
	just_annotated = forms.BooleanField(required = False,initial  = False,label = 'just show annotated docs?')

class VizForm(forms.Form):
	min_degree = forms.IntegerField(required = True,initial = 20,label='Enter minimum topic degree for inclusion')
	edge_thresh = forms.DecimalField(required = True,initial = 0.05,label = 'Enter probability threshold for edge')
	just_annotated_docs = forms.BooleanField(required = False,initial = False,label = 'Just show annotated documents?')
	colour_by_logfc = forms.BooleanField(required = False,initial = False,label = 'colour nodes by logfc')
	discrete_colour = forms.BooleanField(required = False,initial = False,label = 'discrete colouring')
	lower_colour_perc = forms.IntegerField(required = True,initial = 25,label = 'lower colour percentile')
	upper_colour_perc = forms.IntegerField(required = True,initial = 75,label = 'upper colour percentile')

class UserForm(forms.ModelForm):
	password = forms.CharField(widget = forms.PasswordInput())

	class Meta:
		model = User
		fields = ('username','email','password')

class TopicScoringForm(forms.Form):
	upper_perc = forms.IntegerField(required=True,initial=75,label= 'upper percentile')
	lower_perc = forms.IntegerField(required=True,initial=25,label= 'lower percentile')
	storelogfc = forms.BooleanField(initial = False,label = 'Check this box to overwrite stored spectra logfc values')
	def __init__(self,choices,*args,**kwargs):
		super(TopicScoringForm, self).__init__(*args,**kwargs)
		self.fields['group1'] = forms.MultipleChoiceField(choices = choices,label='Pick samples for group 1 (fold change is defined as group 1 over group 2)',required=True)
		self.fields['group2'] = forms.MultipleChoiceField(choices = choices,label='Pick samples for group 2',required=True)