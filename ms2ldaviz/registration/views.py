import logging

from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth import get_user_model
from django.contrib.auth import logout
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic.edit import UpdateView

from basicviz.models import Experiment, UserExperiment
from registration.forms import ProfileForm
from registration.forms import UserForm

User = get_user_model()
logger = logging.getLogger(__name__)


def user_logout(request):
    logout(request)
    return HttpResponseRedirect('/')


def register(request):
    registered = False
    if request.method == 'POST':
        user_form = UserForm(data=request.POST)
        profile_form = ProfileForm(data=request.POST)
        if user_form.is_valid() and profile_form.is_valid():
            # https://www.sitepoint.com/easy-spam-prevention-using-hidden-form-fields/
            if len(request.POST['foo']) == 0:
                user = user_form.save()
                # assign user to Profile first before saving it
                profile = profile_form.save(commit=False)
                profile.user = user
                profile.save()
                add_example_experiments(user)
                registered = True
                messages.success(request, 'Your account has been successfully created')
        else:
            messages.error(request, 'Invalid form entries')
    else:
        user_form = UserForm()
        profile_form = ProfileForm()

    context_dict = {
        'user_form': user_form,
        'profile_form': profile_form,
        'registered': registered
    }
    return render(request, 'registration/register.html', context_dict)


def add_example_experiments(user):
    experiment_list = ['Beer6_POS_IPA_MS1_comparisons',
                       'Urine37_POS_StandardLDA_300Mass2Motifs_MS1peaklist_MS1duplicatefilter',
                       'Beer3_POS_Decomposition_MassBankGNPScombinedset_MS1peaklist_MS1duplicatefilter',
                       'massbank_binned_005',
                       'gnps_binned_005',
                       'Beer3_POS_StandardLDA_300Mass2Motifs_MS1peaklist_MS1duplicatefilter']

    experiments = []
    for ename in experiment_list:
        try:
            e = Experiment.objects.get(name=ename)
            experiments.append(e)
        except:
            print("No such experiment: {}".format(ename))

    for e in experiments:
        ue = UserExperiment.objects.filter(user=user, experiment=e)
        if len(ue) == 0:
            if e.name.startswith('Beer6'):
                UserExperiment.objects.create(user=user, experiment=e, permission='edit')
            else:
                UserExperiment.objects.create(user=user, experiment=e, permission='view')


def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user:
            if user.is_active:
                login(request, user)
                return HttpResponseRedirect('/basicviz/')
            else:
                error_message = 'Account {0} has been disabled'.format(username)
                messages.warning(request, error_message)
                return render(request, 'registration/login.html', {})
        else:
            error_message = 'Invalid login details provided for {0}'.format(username)
            messages.warning(request, error_message)
            return render(request, 'registration/login.html', {})

    else:  # GET
        return render(request, 'registration/login.html', {})


class ProfileUpdate(UpdateView):
    model = User
    template_name = 'registration/user_update.html'
    success_url = reverse_lazy('home')
    fields = ['first_name', 'last_name', 'email']

    def get_object(self, queryset=None):
        return self.request.user
