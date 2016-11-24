from django.conf.urls import patterns, include, url
from annotation import views

urlpatterns = [
	url(r'^$', views.index, name='index'),
	url(r'^start_annotation/(?P<basicviz_experiment_id>\w+)/$',views.start_annotation,name='start_annotation'),
	]