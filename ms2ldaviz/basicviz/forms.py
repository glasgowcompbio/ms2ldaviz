from django import forms
from basicviz.models import Mass2Motif,SystemOptions
from basicviz.constants import AVAILABLE_OPTIONS
from django.contrib.auth.models import User


class Mass2MotifMetadataForm(forms.Form):
    metadata = forms.CharField(max_length = 256,required = False,widget=forms.TextInput(attrs={'size':'80'}))

class Mass2MotifMassbankForm(forms.Form):
    motif_id = forms.CharField(widget=forms.HiddenInput(attrs={'id':'motif_id'}))
    mf_id = forms.CharField(widget=forms.HiddenInput(attrs={'id':'mf_id'}))
    accession = forms.CharField(label='ACCESSION', required=True, widget=forms.TextInput(
        attrs={'id':'accession', 'required':'true', 'size':'50', 'title':'Identifier of the MassBank Record. Two or three letters site code, followed by record id.'})
    )
    authors = forms.CharField(label='AUTHORS', required=True, widget=forms.TextInput(
        attrs={'id':'authors', 'required':'true', 'size':'50', 'title':'Authors and Affiliations of MassBank Record.'})
    )
    comments = forms.CharField(label='COMMENTS', required=False, widget=forms.Textarea(
        attrs={'id': 'comments',
               'rows': '2', 'cols': '50',
               'title': 'In MassBank, COMMENT fields are often used to show the relations of the present record with other MassBank records and with data files.'})
    )
    ch_name = forms.CharField(label='CH$NAME', required=True, widget=forms.Textarea(
        attrs={'id': 'ch_name',
               'rows': '2', 'cols': '50',
               'title': 'Name of the chemical compound analysed.'})
    )
    ch_compound_class = forms.CharField(label='CH$COMPOUND_CLASS', required=True, widget=forms.TextInput(
        attrs={'id':'ch_compound_class', 'required':'true', 'size':'50', 'title':'Category of chemical compound.'})
    )
    ch_formula = forms.CharField(label='CH$FORMULA', required=True, widget=forms.TextInput(
        attrs={'id':'ch_formula', 'required':'true', 'size':'50', 'title':'Molecular formula of chemical compound.'})
    )
    ch_exact_mass = forms.CharField(label='CH$EXACT_MASS', required=True, widget=forms.TextInput(
        attrs={'id':'ch_exact_mass', 'required':'true', 'size':'12', 'title':'Monoisotopic mass of chemical compound.'})
    )
    ch_smiles = forms.CharField(label='CH$SMILES', required=True, widget=forms.TextInput(
        attrs={'id':'ch_smiles', 'required':'true', 'size':'50', 'title':'SMILES string.'})
    )
    ch_iupac = forms.CharField(label='CH$IUPAC', required=True, widget=forms.TextInput(
        attrs={'id':'ch_iupac', 'required':'true', 'size':'50', 'title':'IUPAC International Chemical Identifier (InChI Code).'})
    )
    ch_link = forms.CharField(label='CH$LINK', required=False, widget=forms.Textarea(
        attrs={'id': 'ch_link',
               'rows': '2', 'cols': '50',
               'title': 'Identifier and link of chemical compound to external databases.'})
    )
    ac_instrument = forms.CharField(label='AC$INSTRUMENT', required=True, widget=forms.TextInput(
        attrs={'id':'ac_instrument', 'required':'true', 'size':'50', 'title':'Commercial Name and Model of (Chromatographic Separation Instrument, if any were coupled, and) Mass Spectrometer and Manufacturer.'})
    )
    ac_instrument_type = forms.CharField(label='AC$INSTRUMENT_TYPE', required=True, widget=forms.TextInput(
        attrs={'id':'ac_instrument_type', 'required':'true', 'size':'50', 'title':'General Type of Instrument.'})
    )
    ac_mass_spectrometry_ion_mode = forms.CharField(label='AC$MASS_SPECTROMETRY: ION_MODE', required=True, widget=forms.TextInput(
        attrs={'id':'ac_mass_spectrometry_ion_mode', 'required':'true', 'size':'50'})
    )
    min_rel_int = forms.CharField(label='Min. relative intensity', required=True, widget=forms.TextInput(
        attrs={'id':'min_rel_int', 'required':'true', 'size':'5', 'title':'Minimum relative intensity to filter the features.'})
    )

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
    colour_topic_by_score = forms.BooleanField(required = False,initial = False,label = 'colour motifs by up and down scores')
    discrete_colour = forms.BooleanField(required = False,initial = False,label = 'discrete colouring')
    lower_colour_perc = forms.IntegerField(required = True,initial = 25,label = 'lower colour percentile')
    upper_colour_perc = forms.IntegerField(required = True,initial = 75,label = 'upper colour percentile')
    random_seed = forms.CharField(required = True,initial = 'hello',label = 'seed for network visualisation')
    edge_choice = forms.MultipleChoiceField(required = True, initial = 'probability', choices = (('probability','probability'),('overlap','overlap')),label = 'filter edges by probability or overlap score')

class UserForm(forms.ModelForm):
    password = forms.CharField(widget = forms.PasswordInput())

    class Meta:
        model = User
        fields = ('username','email','password')

class TopicScoringForm(forms.Form):
    upper_perc = forms.IntegerField(required=True,initial=75,label= 'upper percentile')
    lower_perc = forms.IntegerField(required=True,initial=25,label= 'lower percentile')
    storelogfc = forms.BooleanField(initial = False,required = False,label = 'Check this box to overwrite stored spectra logfc values')
    savetopicscores = forms.BooleanField(initial = True,required = False,label = 'Save the computed up and down scores?')
    do_pairs = forms.BooleanField(initial = False, required = False,label = 'Compute scores for topic combinations? (takes a looooong time)')
    def __init__(self,choices,*args,**kwargs):
        super(TopicScoringForm, self).__init__(*args,**kwargs)
        self.fields['group1'] = forms.MultipleChoiceField(choices = choices,label='Pick samples for group 1 (fold change is defined as group 1 over group 2)',required=True)
        self.fields['group2'] = forms.MultipleChoiceField(choices = choices,label='Pick samples for group 2',required=True)

class AlphaDEForm(forms.Form):
    def __init__(self,choices,*args,**kwargs):
        super(AlphaDEForm, self).__init__(*args,**kwargs)
        self.fields['group1'] = forms.MultipleChoiceField(choices = choices,label='Pick samples for group 1 (differential expression is defined as group 1 over group 2)',required=True)
        self.fields['group2'] = forms.MultipleChoiceField(choices = choices,label='Pick samples for group 2',required=True)


class AlphaCorrelationForm(forms.Form):
    edge_thresh = forms.DecimalField(required = True, initial = 0.98,label = 'Enter edge threshold')
    distance_score = forms.ChoiceField(choices = (('cosine','cosine'),('euclidean','euclidean'),('rms','rms'),('pearson','pearson')))
    normalise_alphas = forms.BooleanField(required = False,initial = True,label = 'Normalise alpha vectors?')
    max_edges = forms.IntegerField(required = False,initial = 1000, label = 'Maximum number of edges')
    just_annotated = forms.BooleanField(required = False,initial = False,label = 'Restrict to annotated M2Ms?')

class SystemOptionsForm(forms.ModelForm):
    key_options = [(a[0],a[0]) for a in AVAILABLE_OPTIONS]
    key = forms.ChoiceField(choices = key_options,required = True)
    class Meta:
        model = SystemOptions
        exclude = ('experiment',)
