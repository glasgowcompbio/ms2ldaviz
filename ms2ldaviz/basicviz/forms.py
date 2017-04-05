from django import forms
from django.contrib.auth.models import User

from basicviz.constants import AVAILABLE_OPTIONS
from basicviz.models import SystemOptions, Experiment


class DocFilterForm(forms.Form):
    annotated_only = forms.BooleanField(required=False, label='Restrict to annotated docs?')
    min_annotated_topics = forms.IntegerField(required=False, initial=0,
                                              label='Minimum number of annotated topics per doc')
    topic_threshold = forms.DecimalField(required=False, initial=0.0, label='Probability threshold for topic inclusion')
    only_show_annotated = forms.BooleanField(required=False, initial=False, label='Only display annotated topics')


class ValidationForm(forms.Form):
    p_thresh = forms.DecimalField(required=True, initial=0.05,
                                  label='Enter document -> mass2motif membership threshold')
    just_annotated = forms.BooleanField(required=False, initial=False, label='just show annotated docs?')


class VizForm(forms.Form):
    min_degree = forms.IntegerField(required=True, initial=20, label='Enter minimum topic degree for inclusion')
    edge_thresh = forms.DecimalField(required=True, initial=0.05, label='Enter probability threshold for edge')
    just_annotated_docs = forms.BooleanField(required=False, initial=False, label='Just show annotated documents?')
    colour_by_logfc = forms.BooleanField(required=False, initial=False, label='colour nodes by logfc')
    colour_topic_by_score = forms.BooleanField(required=False, initial=False,
                                               label='colour motifs by up and down scores')
    discrete_colour = forms.BooleanField(required=False, initial=False, label='discrete colouring')
    lower_colour_perc = forms.IntegerField(required=True, initial=25, label='lower colour percentile')
    upper_colour_perc = forms.IntegerField(required=True, initial=75, label='upper colour percentile')
    random_seed = forms.CharField(required=True, initial='hello', label='seed for network visualisation')
    edge_choice = forms.MultipleChoiceField(required=True, initial='probability',
                                            choices=(('probability', 'probability'), ('overlap_score', 'overlap_score')),
                                            label='filter edges by probability or overlap score')


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
    other_experiment = forms.ChoiceField(choices = [(e.id,e.name) for e in Experiment.objects.all()],required = True)
    min_score_to_save = forms.FloatField(required = True,initial = 0.5)