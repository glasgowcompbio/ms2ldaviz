from django import forms
from django.contrib.auth.models import User
from django.db.models import Q

from motifdb.models import MDBMotifSet
from basicviz.models import Experiment,BVFeatureSet,UserExperiment,PublicExperiments

class MatchMotifDBForm(forms.Form):
    motif_set = forms.ModelChoiceField(queryset = MDBMotifSet.objects.all(),required=True)
    min_score_to_save = forms.FloatField(required = True,initial=0.5)


class NewMotifSetForm(forms.Form):
    motifset_name = forms.CharField(max_length=1024,required = True)
    motif_name_prefix = forms.CharField(max_length=10,required = True)
    description = forms.CharField(required=True,widget = forms.Textarea)
    ms2lda_experiment = forms.ModelChoiceField(queryset=Experiment.objects.none(),
                                              required=False, label="Create from MS2LDA experiment")
    
    ionization = forms.ChoiceField(required = False,choices = (
        ("electrospray ionization (+ve)","electrospray ionization (+ve)"),
        ("electrospray ionization (+ve)","electrospray ionization (+ve)"),
    )) 
    
    # only show this if ms2lda_experiment isn't selected
    # bin_width = forms.DecimalField(required = False)

    mass_spectrometer = forms.ChoiceField(required = True, choices = (
        ("Maxis_Impact","Maxis_Impact"), 
        ("Maxis_ImpactHD","Maxis_ImpactHD"), 
        ("QExactive","QExactive"), 
        ("micrOTOF-Q II","micrOTOF-Q II"),
        ("Ion Trap","Ion Trap"), 
        ("Orbitrap","Orbitrap"), 
        ("other high resolution instrument","other high resolution instrument"), 
        ("other low resolution instrument","other low resolution instrument"), 
        ("mixture","mixture"),
    ))

    collision_energy = forms.ChoiceField(required = True, choices = (
        ("HCD","HCD"),
        ("CID","CID"),
        ("mixture","mixture"),
    ))

    taxon_id = forms.CharField(required = False)

    scientific_name = forms.CharField(required = False)

    sample_type = forms.CharField(required = False)

    paper_url = forms.CharField(required = False)

    chromatography = forms.ChoiceField(required = False, choices = (
        ("reverse phase (C18)","reverse phase (C18)"),
        ("reverse phase (C8)","reverse phase (C8)"),
        ("reverse phase (Phenyl-Hexyl)","reverse phase (Phenyl-Hexyl)"),
        ("normal phase (HILIC)","normal phase (HILIC)"), 
        ("mixed mode (Scherzo SM-C18, direct infusion (DI)","mixed mode (Scherzo SM-C18, direct infusion (DI)"),
    ))

    other_information = forms.CharField(widget=forms.Textarea,required= False)
    massive_id = forms.CharField(required = False)



    def __init__(self, user, *args, **kwargs):
        super(NewMotifSetForm, self).__init__(*args, **kwargs)

        fs = BVFeatureSet.objects.filter(name__in = ['binned_005','binned_01','binned_1'])
        ue = UserExperiment.objects.filter(user = user)
        pe = PublicExperiments.objects.all()
        experiments = Experiment.objects.filter(Q(featureset__in = fs), 
            (Q(id__in = [i.experiment.id for i in ue]) | Q(id__in = [p.experiment.id for p in pe])))
        self.fields['ms2lda_experiment'].queryset = experiments

class ChooseMotifs(forms.Form):
    motifs = forms.MultipleChoiceField(choices = (('a','a')),widget=forms.SelectMultiple())
    def __init__(self, motifs, *args, **kwargs):
        super(ChooseMotifs, self).__init__(*args, **kwargs)
        self.fields['motifs'].choices = motifs

