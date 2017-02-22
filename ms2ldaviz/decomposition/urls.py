# urls.py for decomposition
from django.conf.urls import include, url
from decomposition import views

urlpatterns = [
	url(r'^view_parents/(?P<experiment_id>\w+)/(?P<mass2motif_id>\w+)/$', views.view_parents, name='view_parents_decomposition'),
	url(r'^get_parents/(?P<experiment_id>\w+)/(?P<mass2motif_id>\w+)/$', views.get_parents, name='get_parents_decomposition'),
	url(r'^get_word_graph/(?P<mass2motif_id>\w+)/(?P<vo_id>\w+)/(?P<experiment_id>\w+)/$', views.get_word_graph, name='get_word_graph'),
	url(r'^get_intensity/(?P<mass2motif_id>\w+)/(?P<vo_id>\w+)/(?P<experiment_id>\w+)/$', views.get_intensity, name='get_intensity'),
	]
