from django.conf.urls import include, url
from motifdb import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^motif_set/(?P<motif_set_id>\w+)/$', views.motif_set,name='motif_set'),
    url(r'^motif/(?P<motif_id>\w+)/$', views.motif,name='motif'),
    url(r'^start_motif_matching/(?P<experiment_id>\w+)/$', views.start_motif_matching,name='start_motif_matching'),
]
