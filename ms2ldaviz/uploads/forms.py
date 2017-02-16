from django import forms

from basicviz.models import Experiment


class CreateExperimentForm(forms.ModelForm):

    class Meta:
        model = Experiment
        widgets = {
            'name': forms.TextInput(attrs={'style': 'width:300px'}),
            'description': forms.Textarea(attrs={'rows': 4, 'cols': 100}),
        }
        exclude = ('status',)