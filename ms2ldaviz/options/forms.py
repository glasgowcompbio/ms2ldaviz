from django import forms

from options.constants import AVAILABLE_OPTIONS
from basicviz.models import SystemOptions


class SystemOptionsForm(forms.ModelForm):
    key_options = [(a[0], a[0]) for a in AVAILABLE_OPTIONS]
    key = forms.ChoiceField(choices=key_options, required=True)

    class Meta:
        model = SystemOptions
        exclude = ('experiment',)
