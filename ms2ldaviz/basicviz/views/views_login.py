from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render

from basicviz.forms import UserForm
from basicviz.models import UserExperiment, ExtraUsers, \
    MultiFileExperiment, MultiLink


@login_required(login_url='/basicviz/login/')
def index(request):
    userexperiments = UserExperiment.objects.filter(user=request.user)
    experiments = []
    for ue in userexperiments:
        experiments.append(ue.experiment)

    # Remove those that are multi ones
    exclude_individuals = []

    for experiment in experiments:
        links = MultiLink.objects.filter(experiment=experiment)
        if len(links) > 0:
            exclude_individuals += [l.experiment for l in links]

    print exclude_individuals
    for e in exclude_individuals:
        del experiments[experiments.index(e)]

    experiments = list(set(experiments))

    # experiments = Experiment.objects.all()
    context_dict = {'experiments': experiments}
    context_dict['user'] = request.user
    eu = ExtraUsers.objects.filter(user=request.user)

    mfe = MultiFileExperiment.objects.all()

    if len(eu) > 0:
        extra_user = True
    else:
        extra_user = False
    context_dict['extra_user'] = extra_user
    context_dict['mfe'] = mfe
    return render(request, 'basicviz/basicviz.html', context_dict)


@login_required(login_url='/basicviz/login/')
def user_logout(request):
    # Since we know the user is logged in, we can now just log them out.
    logout(request)

    # Take the user back to the homepage.
    return HttpResponseRedirect('/')


def register(request):
    registered = False
    if request.method == 'POST':
        user_form = UserForm(data=request.POST)

        if user_form.is_valid():
            user = user_form.save()
            user.set_password(user.password)
            user.save()

            registered = True

        else:
            print user_form.errors

    else:
        user_form = UserForm()

    context_dict = {'user_form': user_form, 'registered': registered}
    return render(request,
                  'basicviz/register.html', context_dict)


def user_login(request):
    # If the request is a HTTP POST, try to pull out the relevant information.
    if request.method == 'POST':
        # Gather the username and password provided by the user.
        # This information is obtained from the login form.
        # We use request.POST.get('<variable>') as opposed to request.POST['<variable>'],
        # because the request.POST.get('<variable>') returns None, if the value does not exist,
        # while the request.POST['<variable>'] will raise key error exception
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Use Django's machinery to attempt to see if the username/password
        # combination is valid - a User object is returned if it is.
        user = authenticate(username=username, password=password)

        # If we have a User object, the details are correct.
        # If None (Python's way of representing the absence of a value), no user
        # with matching credentials was found.
        if user:
            # Is the account active? It could have been disabled.
            if user.is_active:
                # If the account is valid and active, we can log the user in.
                # We'll send the user back to the homepage.
                login(request, user)
                return HttpResponseRedirect('/basicviz/')
            else:
                # An inactive account was used - no logging in!
                return HttpResponse("Your account is disabled.")
        else:
            # Bad login details were provided. So we can't log the user in.
            print "Invalid login details: {0}, {1}".format(username, password)
            return HttpResponse("Invalid login details supplied.")

    # The request is not a HTTP POST, so display the login form.
    # This scenario would most likely be a HTTP GET.
    else:
        # No context variables to pass to the template system, hence the
        # blank dictionary object...
        return render(request, 'basicviz/login.html', {})
