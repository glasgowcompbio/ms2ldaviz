from django.shortcuts import render
from django.template.loader import render_to_string


def home(request):
    context_dict = {}
    return render(request,'ms2ldaviz/index.html',context_dict)


def people(request):
    context_dict = {}
    return render(request,'ms2ldaviz/people.html',context_dict)


def api(request):
    context_dict = {}
    return render(request,'ms2ldaviz/api.html',context_dict)


def user_guide(request):
    user_guide_str = render_to_string('user_guide/user_guide.md')
    return render(request, 'user_guide/user_guide.html', {'user_guide_str':user_guide_str})