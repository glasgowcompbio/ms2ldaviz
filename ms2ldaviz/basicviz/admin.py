from django.contrib import admin
from basicviz.models import Experiment,Document,Mass2Motif,Feature,FeatureMass2MotifInstance,VizOptions,UserExperiment,ExtraUsers,MultiFileExperiment,MultiLink,Alpha,PeakSet,IntensityInstance


admin.site.register(Experiment)
admin.site.register(Document)
admin.site.register(Mass2Motif)
admin.site.register(Feature)
admin.site.register(FeatureMass2MotifInstance)
admin.site.register(VizOptions)
admin.site.register(UserExperiment)
admin.site.register(ExtraUsers)
admin.site.register(MultiFileExperiment)
admin.site.register(MultiLink)
admin.site.register(Alpha)
admin.site.register(PeakSet)
admin.site.register(IntensityInstance)