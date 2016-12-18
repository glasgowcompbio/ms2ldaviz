from django.contrib import admin

# Register your models here.
from annotation.models import SubstituentTerm,TaxaTerm,SubstituentInstance,TaxaInstance

admin.site.register(SubstituentTerm)
admin.site.register(SubstituentInstance)
admin.site.register(TaxaInstance)
admin.site.register(TaxaTerm)