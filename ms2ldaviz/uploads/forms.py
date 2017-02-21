from django import forms

from basicviz.models import Experiment


class CreateExperimentForm(forms.ModelForm):

    class Meta:
        model = Experiment
        widgets = {
            'name': forms.TextInput(attrs={'style': 'width:300px'}),
            'description': forms.Textarea(attrs={'rows': 4, 'cols': 100}),
            'csv_file': forms.ClearableFileInput(),
            'mzml_file': forms.ClearableFileInput()
        }
        labels = {
            'csv_file': 'MS1 File (CSV)',
            'mzml_file': 'MS2 File (mzML)'
        }
        exclude = ('status',)