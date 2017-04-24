from django import forms

from basicviz.models import Experiment
from decomposition.models import MotifSet
from basicviz.constants import EXPERIMENT_DECOMPOSITION_SOURCE
from django.core.exceptions import ValidationError
from django.utils.translation import ugettext_lazy as _


class CreateExperimentForm(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        super(CreateExperimentForm, self).__init__(*args, **kwargs)
        self.fields['decompose_from'] = forms.ModelChoiceField(
            queryset=MotifSet.objects.all(),
            label='Decompose using Mass2Motifs in'
        )
        self.fields['ms2_file'].required = True
        self.fields['decompose_from'].required = False

    def is_valid(self):
      valid = super(CreateExperimentForm, self).is_valid()
      if not valid:
        return valid

      ms2_format = self.cleaned_data['experiment_ms2_format']
      ms2_file_name = self.cleaned_data['ms2_file'].name.lower()
      if ms2_format == '0':
        if not ms2_file_name.endswith('.mzml'):
          self.add_error('ms2_file', ValidationError(_('Extension should be .mzML'), code='invalid'))
          return False
      elif ms2_format == '1':
        if not ms2_file_name.endswith('.msp'):
          self.add_error('ms2_file', ValidationError(_('Extension should be .msp'), code='invalid'))
          return False
      elif ms2_format == '2':
        if not ms2_file_name.endswith('.mgf'):
          self.add_error('ms2_file', ValidationError(_('Extension should be .mgf'), code='invalid'))
          return False

      return True

    class Meta:
        model = Experiment
        widgets = {
            'name': forms.TextInput(attrs={'style': 'width:300px'}),
            'description': forms.Textarea(attrs={'rows': 4, 'cols': 100}),
            'csv_file': forms.ClearableFileInput(),
            'ms2_file': forms.ClearableFileInput()
        }
        labels = {
            'name': '(Required) Experiment name. Note that this must be unique in the system',
            'description':'(Required) Experiment description.',
            'csv_file': 'MS1 file (CSV) [see below for formatting instructions]',
            'ms2_file': '(Required) MS2 file (mzML, msp, or mgf)',
            'isolation_window': 'Fragmentation isolation window. Used to match fragment spectra with MS1 peaks.',
            'mz_tol': 'Mass tolerance when linking peaks from the peaklist to those found in the mzML (ppm)',
            'rt_tol': 'Retention time tolerance when peaks from the peaklist to those found in the mzML (seconds)',
            'min_ms1_rt': 'Minimum retention time of MS1 peaks to store (seconds)',
            'max_ms1_rt': 'Maximum retention time of MS1 peaks to store (seconds)',
            'min_ms1_intensity': 'Minimum intensity of MS1 peaks to store',
            'min_ms2_intensity': 'Minimum intensity of MS2 peaks to store',
            'filter_duplicates': 'Attempt to filter out duplicate MS1 peaks. If Set to True, the code merges peaks within duplicate_filter_mz_tol and duplicate_filter_rt_tol.',
            'duplicate_filter_mz_tol': 'mz tol (ppm) for duplicate filtering',
            'duplicate_filter_rt_tol': 'rt tol (seconds) for duplicate filtering',
            'K': 'Number of Mass2Motifs',
            'decomposition_source': 'Use for decomposition in the future?',
            'n_its': 'Number of iterations (for LDA).',
            'experiment_type': '(Required) LDA, or decomposition,'
        }
        fields = [
            'name', 'description',
            'experiment_type', 'experiment_ms2_format', 'ms2_file', 'csv_file',
            'isolation_window', 'mz_tol', 'rt_tol', 'min_ms1_rt', 'max_ms1_rt', 'min_ms1_intensity','min_ms2_intensity',
            'filter_duplicates','duplicate_filter_mz_tol','duplicate_filter_rt_tol',
            'K', 'n_its',
        ]
        exclude = ('status',)
