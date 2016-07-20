from django.shortcuts import render

def home(request):
	context_dict = {}
	return render(request,'ms2ldaviz/index.html',context_dict)