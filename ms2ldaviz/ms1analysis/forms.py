from django import forms
from django.contrib.auth.models import User

from basicviz.models import Experiment
from ms1analysis.models import Analysis

class AnalysisForm(forms.ModelForm):
    def __init__(self, choices, *args, **kwargs):
        super(AnalysisForm, self).__init__(*args, **kwargs)
        self.fields['group1'] = forms.MultipleChoiceField(choices=choices,
                                                          widget=forms.SelectMultiple,
                                                          label='Pick samples for group 1 (use SHIFT to select multiple samples)',
                                                          required=True)
        self.fields['group2'] = forms.MultipleChoiceField(choices=choices,
                                                          widget=forms.SelectMultiple,
                                                          label='Pick samples for group 2 (use SHIFT to select multiple samples)',
                                                          required=True)
        self.fields['group1'].required = True
        self.fields['group2'].required = True
        # self.fields['use_logarithm'].required = True

    class Meta:
        model = Analysis
        widgets = {
            'name': forms.TextInput(attrs={'style': 'width:300px'}),
            'description': forms.Textarea(attrs={'rows': 4, 'cols': 100})
        }
        labels = {
        }
        fields = [
            'name', 'description', 'group1', 'group2'
        ]