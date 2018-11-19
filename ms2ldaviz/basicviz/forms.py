from django import forms
from django.contrib.auth.models import User
from django.db.models import Q

from basicviz.constants import AVAILABLE_OPTIONS
from basicviz.models import SystemOptions, Experiment, UserExperiment, BVFeatureSet, PublicExperiments
from motifdb.models import MDBMotifSet


class DocFilterForm(forms.Form):
    annotated_only = forms.BooleanField(required=False, label='Restrict to annotated docs?')
    min_annotated_topics = forms.IntegerField(required=False, initial=0,
                                              label='Minimum number of annotated topics per doc')
    topic_threshold = forms.DecimalField(required=False, initial=0.0, label='Probability threshold for topic inclusion')
    only_show_annotated = forms.BooleanField(required=False, initial=False, label='Only display annotated topics')


class ValidationForm(forms.Form):
    p_thresh = forms.DecimalField(required=True, initial=0.05,
                                  label='Enter probability threshold')
    overlap_thresh = forms.DecimalField(required=True, initial=0.0,
                                  label='Enter overlap threshold')
    just_annotated = forms.BooleanField(required=False, initial=False, label='just show annotated docs?')


class VizForm(forms.Form):
    min_degree = forms.IntegerField(required=True, initial=5, label='Enter minimum topic degree for inclusion')
    # edge_thresh = forms.DecimalField(required=True, initial=0.05, label='Enter probability threshold for edge')
    # just_annotated_docs = forms.BooleanField(required=False, initial=False, label='Just show annotated documents?')
    # colour_by_logfc = forms.BooleanField(required=False, initial=False, label='colour nodes by logfc')
    # colour_topic_by_score = forms.BooleanField(required=False, initial=False,
    #                                            label='colour motifs by up and down scores')
    # discrete_colour = forms.BooleanField(required=False, initial=False, label='discrete colouring')
    # lower_colour_perc = forms.IntegerField(required=True, initial=25, label='lower colour percentile')
    # upper_colour_perc = forms.IntegerField(required=True, initial=75, label='upper colour percentile')
    # random_seed = forms.CharField(required=True, initial='hello', label='seed for network visualisation')
    # edge_choice = forms.MultipleChoiceField(required=True, initial='probability',
    #                                         choices=(('probability', 'probability'), ('overlap_score', 'overlap_score')),
    #                                         label='filter edges by probability or overlap score')
    ms1_analysis = forms.MultipleChoiceField(choices=(),
                                                  label='Pick one MS1 analysis',
                                                  required=False)

    def __init__(self, choices, *args, **kwargs):
        super(VizForm, self).__init__(*args, **kwargs)
        ## add default empty choice
        self.fields['ms1_analysis'].choices = [('', '----------')] + choices

class TopicScoringForm(forms.Form):
    upper_perc = forms.IntegerField(required=True, initial=75, label='upper percentile')
    lower_perc = forms.IntegerField(required=True, initial=25, label='lower percentile')
    storelogfc = forms.BooleanField(initial=False, required=False,
                                    label='Check this box to overwrite stored spectra logfc values')
    savetopicscores = forms.BooleanField(initial=True, required=False, label='Save the computed up and down scores?')
    do_pairs = forms.BooleanField(initial=False, required=False,
                                  label='Compute scores for topic combinations? (takes a looooong time)')

    def __init__(self, choices, *args, **kwargs):
        super(TopicScoringForm, self).__init__(*args, **kwargs)
        self.fields['group1'] = forms.MultipleChoiceField(choices=choices,
                                                          label='Pick samples for group 1 (fold change is defined as group 1 over group 2)',
                                                          required=True)
        self.fields['group2'] = forms.MultipleChoiceField(choices=choices, label='Pick samples for group 2',
                                                          required=True)


class AlphaDEForm(forms.Form):
    def __init__(self, choices, *args, **kwargs):
        super(AlphaDEForm, self).__init__(*args, **kwargs)
        self.fields['group1'] = forms.MultipleChoiceField(choices=choices,
                                                          label='Pick samples for group 1 (differential expression is defined as group 1 over group 2)',
                                                          required=True)
        self.fields['group2'] = forms.MultipleChoiceField(choices=choices, label='Pick samples for group 2',
                                                          required=True)


class AlphaCorrelationForm(forms.Form):
    edge_thresh = forms.DecimalField(required=True, initial=0.98, label='Enter edge threshold')
    distance_score = forms.ChoiceField(
        choices=(('cosine', 'cosine'), ('euclidean', 'euclidean'), ('rms', 'rms'), ('pearson', 'pearson')))
    normalise_alphas = forms.BooleanField(required=False, initial=True, label='Normalise alpha vectors?')
    max_edges = forms.IntegerField(required=False, initial=1000, label='Maximum number of edges')
    just_annotated = forms.BooleanField(required=False, initial=False, label='Restrict to annotated M2Ms?')


class MatchMotifForm(forms.Form):

    other_experiment = forms.ModelChoiceField(queryset=Experiment.objects.none(),
                                              required=True, label="Match against")
    min_score_to_save = forms.FloatField(required=True, initial=0.5)

    def __init__(self, user, *args, **kwargs):
        super(MatchMotifForm, self).__init__(*args, **kwargs)

        # Select only the experiments that belong to this user (through UserExperiment)
        # and also not a multi-file experiment (through MultiLink), because there are too many of them
        # self.fields['other_experiment'].queryset = Experiment.objects.filter(
        #     userexperiment__user=user, multilink__isnull=True).order_by('name')
        # Modified by SR to include the multifile ones - 11/7/17
        # Modified by SR again to include only those with a featureset

        # modified by SR, 15/10/18 to list those from the stated featuresets that either
        # are accessible by the user, or are public
        fs = BVFeatureSet.objects.filter(name__in = ['binned_005','binned_01','binned_1'])
        ue = UserExperiment.objects.filter(user = user)
        pe = PublicExperiments.objects.all()
        experiments = Experiment.objects.filter(Q(featureset__in = fs), 
            (Q(id__in = [i.experiment.id for i in ue]) | Q(id__in = [p.experiment.id for p in pe])))

        # print len(experiments)
        # experiments = experiments.filter(userexperiment__user = user) | experiments.filter(publicexperiments )
        # print len(experiments)
        # users_experiments += [p.experiment for p in PublicExperiments.objects.all()]
        # experiments = users_experiments
        # for e in users_experiments:
        #     if e.featureset in fs:
        #         experiments.append(e)
        # # experiments = Experiment.objects.filter(featureset__in = fs).order_by('name')
        # self.fields['other_experiment'].queryset = Experiment.objects.filter(
        #     userexperiment__user=user).order_by('name')
        self.fields['other_experiment'].queryset = experiments

