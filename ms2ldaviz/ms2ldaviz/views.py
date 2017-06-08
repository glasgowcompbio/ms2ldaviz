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
    markdown_str = render_to_string('markdowns/user_guide.md')
    return render(request, 'markdowns/user_guide.html', {'markdown_str':markdown_str})


def disclaimer(request):
    markdown_str = render_to_string('markdowns/disclaimer.md')
    return render(request, 'markdowns/disclaimer.html', {'markdown_str':markdown_str})


def confidence(request):
    markdown_str = render_to_string('markdowns/confidence.md')
    return render(request, 'markdowns/confidence.html', {'markdown_str':markdown_str})