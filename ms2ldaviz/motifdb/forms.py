from django import forms
from django.contrib.auth.models import User
from django.db.models import Q

from motifdb.models import MDBMotifSet
from basicviz.models import Experiment,BVFeatureSet,UserExperiment,PublicExperiments

class MatchMotifDBForm(forms.Form):
    motif_set = forms.ModelChoiceField(queryset = MDBMotifSet.objects.all(),required=True)
    min_score_to_save = forms.FloatField(required = True,initial=0.5)


class NewMotifSetForm(forms.Form):
    motifset_name = forms.CharField(max_length=1024,required = True, label = "Name")
    Motif_Name_Prefix = forms.CharField(max_length=10,required = True, label = "Motif Name Prefix")
    description = forms.CharField(required=True,widget = forms.Textarea)
    ms2lda_experiment = forms.ModelChoiceField(queryset=Experiment.objects.none(),
                                              required=False, label="Create from MS2LDA experiment")
    
    Analysis_Polarity = forms.ChoiceField(required = True,label = "Analysis_Polarity", choices = (
        ("positive ionisation mode","positive ionisation mode"),
        ("negative ionisation mode","negative ionisation mode"),
    )) 
    
    Analysis_IonizationSources_ionization_source = forms.ChoiceField(required = True,label = "Analysis_IonizationSource", choices = (
        ("electospray ionization","electospray ionization"),
        ("other","other"),
    ))
    # only show this if ms2lda_experiment isn't selected
    # bin_width = forms.DecimalField(required = False)

    Analysis_MassSpectrometer = forms.ChoiceField(required = True, label = "Analysis_MassSpectrometer",choices = (
        ("Maxis_Impact","Maxis_Impact"), 
        ("Maxis_ImpactHD","Maxis_ImpactHD"), 
        ("QExactive","QExactive"), 
        ("micrOTOF-Q II","micrOTOF-Q II"),
        ("Ion Trap","Ion Trap"), 
        ("Orbitrap","Orbitrap"), 
        ("Time-Of-Flight (ToF)","Time-Of-Flight (ToF)"),
        ("other high resolution instrument","other high resolution instrument"), 
        ("other low resolution instrument","other low resolution instrument"), 
        ("mixture","mixture"),
    ))

    Collision_Energy = forms.ChoiceField(label = "Collision_Energy",required = True, choices = (
        ("HCD","HCD"),
        ("CID","CID"),
        ("mixture","mixture"),
    ))

    Taxon_ID = forms.CharField(label = "Taxon_ID",required = False)

    Scientific_Name = forms.CharField(label = "Scientific_Name",required = False)

    Sample_Type = forms.CharField(label = "Sample_Type",required = False)

    Paper_URL = forms.CharField(label = "Paper_URL",required = False)

    Analysis_ChromatographyAndPhase = forms.ChoiceField(required = False, label = "Analysis_ChromatographyAndPhase", choices = (
        ("reverse phase (C18)","reverse phase (C18)"),
        ("reverse phase (C8)","reverse phase (C8)"),
        ("reverse phase (Phenyl-Hexyl)","reverse phase (Phenyl-Hexyl)"),
        ("normal phase (HILIC)","normal phase (HILIC)"), 
        ("mixed mode (Scherzo SM-C18)","mixed mode (Scherzo SM-C18)"),
        ("direct infusion (DI)","direct infusion (DI)"),
    ))

    Other_Information = forms.CharField(label = "Other_Information",widget=forms.Textarea,required= False)
    Massive_ID = forms.CharField(label = "Massive_ID",required = False)



    def __init__(self, user, *args, **kwargs):
        super(NewMotifSetForm, self).__init__(*args, **kwargs)

        fs = BVFeatureSet.objects.filter(name__in = ['binned_005','binned_01','binned_1'])
        ue = UserExperiment.objects.filter(user = user)
        pe = PublicExperiments.objects.all()
        experiments = Experiment.objects.filter(Q(featureset__in = fs), 
            (Q(id__in = [i.experiment.id for i in ue]) | Q(id__in = [p.experiment.id for p in pe])))
        self.fields['ms2lda_experiment'].queryset = experiments


class MetadataForm(forms.Form):
    motifset_name = forms.CharField(max_length=1024,required = True, label = "Name")
    # motif_name_prefix = forms.CharField(max_length=10,required = True, label = "Motif Name Prefix")
    description = forms.CharField(required=True,widget = forms.Textarea)
    # ms2lda_experiment = forms.ModelChoiceField(queryset=Experiment.objects.none(),
                                            #   required=False, label="Create from MS2LDA experiment")
    
    Analysis_Polarity = forms.ChoiceField(required = True,label = "Analysis_Polarity", choices = (
        ("positive ionisation mode","positive ionisation mode"),
        ("negative ionisation mode","negative ionisation mode"),
    )) 
    
    Analysis_IonizationSource = forms.ChoiceField(required = True,label = "Analysis_IonizationSource", choices = (
        ("electospray ionization","electospray ionization"),
        ("other","other"),
    ))
    # only show this if ms2lda_experiment isn't selected
    # bin_width = forms.DecimalField(required = False)

    Analysis_MassSpectrometer = forms.ChoiceField(required = True, label = "Analysis_MassSpectrometer",choices = (
        ("Maxis_Impact","Maxis_Impact"), 
        ("Maxis_ImpactHD","Maxis_ImpactHD"), 
        ("QExactive","QExactive"), 
        ("micrOTOF-Q II","micrOTOF-Q II"),
        ("Ion Trap","Ion Trap"), 
        ("Orbitrap","Orbitrap"), 
        ("Time-Of-Flight (ToF)","Time-Of-Flight (ToF)"),
        ("other high resolution instrument","other high resolution instrument"), 
        ("other low resolution instrument","other low resolution instrument"), 
        ("mixture","mixture"),
    ))

    Collision_Energy = forms.ChoiceField(required = True, choices = (
        ("HCD","HCD"),
        ("CID","CID"),
        ("mixture","mixture"),
    ))

    Taxon_ID = forms.CharField(label = "Taxon_ID",required = False)

    Scientific_Name = forms.CharField(required = False)

    Sample_Type = forms.CharField(required = False)

    Paper_URL = forms.CharField(label = "Paper_URL",required = False)

    Analysis_ChromatographyAndPhase = forms.ChoiceField(required = False, label = "Analysis_ChromatographyAndPhase", choices = (
        ("reverse phase (C18)","reverse phase (C18)"),
        ("reverse phase (C8)","reverse phase (C8)"),
        ("reverse phase (Phenyl-Hexyl)","reverse phase (Phenyl-Hexyl)"),
        ("normal phase (HILIC)","normal phase (HILIC)"), 
        ("mixed mode (Scherzo SM-C18)","mixed mode (Scherzo SM-C18)"),
        ("direct infusion (DI)","direct infusion (DI)"),
    ))

    Other_Information = forms.CharField(widget=forms.Textarea,required= False)
    Massive_ID = forms.CharField(label = "Massive_ID",required = False)



    # def __init__(self, user, *args, **kwargs):
    #     super(NewMotifSetForm, self).__init__(*args, **kwargs)

    #     fs = BVFeatureSet.objects.filter(name__in = ['binned_005','binned_01','binned_1'])
    #     ue = UserExperiment.objects.filter(user = user)
    #     pe = PublicExperiments.objects.all()
    #     experiments = Experiment.objects.filter(Q(featureset__in = fs), 
    #         (Q(id__in = [i.experiment.id for i in ue]) | Q(id__in = [p.experiment.id for p in pe])))
    #     self.fields['ms2lda_experiment'].queryset = experiments


class ChooseMotifs(forms.Form):
    motifs = forms.MultipleChoiceField(choices = (('a','a')),widget=forms.SelectMultiple())
    def __init__(self, motifs, *args, **kwargs):
        super(ChooseMotifs, self).__init__(*args, **kwargs)
        self.fields['motifs'].choices = motifs

