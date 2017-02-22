# urls.py for decomposition
from django.conf.urls import include, url
from decomposition import views

urlpatterns = [
	url(r'^view_parents/(?P<experiment_id>\w+)/(?P<mass2motif_id>\w+)/$', views.view_parents, name='view_parents_decomposition'),
	url(r'^get_parents/(?P<experiment_id>\w+)/(?P<mass2motif_id>\w+)/$', views.get_parents, name='get_parents_decomposition'),
	]
