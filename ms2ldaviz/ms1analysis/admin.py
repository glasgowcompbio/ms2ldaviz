from django.contrib import admin

from ms1analysis.models import Sample, DocSampleIntensity, Analysis, AnalysisResult, AnalysisResultPlage

admin.site.register(Sample)
admin.site.register(DocSampleIntensity)
admin.site.register(Analysis)
admin.site.register(AnalysisResult)
admin.site.register(AnalysisResultPlage)