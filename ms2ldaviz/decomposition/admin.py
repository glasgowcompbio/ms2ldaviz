from django.contrib import admin

# Register your models here.

from decomposition.models import Decomposition,FeatureSet,GlobalFeature,GlobalMotif,FeatureMap,Beta,DocumentGlobalFeature,MotifSet

admin.site.register(Decomposition)
admin.site.register(FeatureSet)
admin.site.register(GlobalFeature)
admin.site.register(GlobalMotif)
admin.site.register(FeatureMap)
admin.site.register(Beta)
admin.site.register(DocumentGlobalFeature)
admin.site.register(MotifSet)