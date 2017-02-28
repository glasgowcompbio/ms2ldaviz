from django import forms

from basicviz.models import Experiment
from decomposition.models import MotifSet
from basicviz.constants import EXPERIMENT_DECOMPOSITION_SOURCE


class CreateExperimentForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        super(CreateExperimentForm, self).__init__(*args, **kwargs)
        self.fields['decompose_from'] = forms.ModelChoiceField(
            queryset=MotifSet.objects.all(),
            label='Decompose using Mass2Motifs in'
        )
        self.fields['mzml_file'].required = True
        self.fields['decompose_from'].required = False

    class Meta:
        model = Experiment
        widgets = {
            'name': forms.TextInput(attrs={'style': 'width:300px'}),
            'description': forms.Textarea(attrs={'rows': 4, 'cols': 100}),
            'csv_file': forms.ClearableFileInput(),
            'mzml_file': forms.ClearableFileInput()
        }
        labels = {
            'csv_file': 'MS1 file (CSV)',
            'mzml_file': 'MS2 file (mzML)',
            'isolation_window': 'Isolation window when linking MS1-MS2 peaks (Da)',
            'mz_tol': 'Mass tolerance when linking MS1-MS2 peaks (ppm)',
            'rt_tol': 'Retention time tolerance when linking MS1-MS2 peaks (seconds)',
            'min_ms1_rt': 'Minimum retention time of MS1 peaks to keep (seconds)',
            'max_ms1_rt': 'Maximum retention time of MS1 peaks to keep (seconds)',
            'min_ms2_intensity': 'Minimum intensity of MS2 peaks to keep',
            'K': 'Number of Mass2Motifs',
            'decomposition_source': 'Use for decomposition in the future?',
        }
        fields = [
            'name', 'description',
            'experiment_type', 'csv_file', 'mzml_file',
            'isolation_window', 'mz_tol', 'rt_tol', 'min_ms1_rt', 'max_ms1_rt', 'min_ms2_intensity',
            'K', 'decomposition_source',
        ]
        exclude = ('status',)