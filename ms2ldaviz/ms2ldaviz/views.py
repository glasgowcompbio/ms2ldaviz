from django.shortcuts import render

def home(request):
    context_dict = {}
    return render(request,'ms2ldaviz/index.html',context_dict)

def people(request):
    context_dict = {}
    return render(request,'ms2ldaviz/people.html',context_dict)

def api(request):
    context_dict = {}
    return render(request,'ms2ldaviz/api.html',context_dict)