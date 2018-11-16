from django.conf.urls import include, url
from motifdb import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^motif_set/(?P<motif_set_id>\w+)/$', views.motif_set,name='motif_set'),
]
