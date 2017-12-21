from django import forms

class AnalysisForm(forms.Form):
    name = forms.CharField(widget=forms.TextInput(attrs={'style': 'width:300px'}))
    description = forms.CharField(widget=forms.Textarea(attrs={'rows': 4, 'cols': 80}))
    use_normalization = forms.ChoiceField(choices=[('N', 'No'), ('Y', 'Yes')], required=True)
