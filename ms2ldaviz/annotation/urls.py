from django.conf.urls import patterns, include, url
from annotation import views

urlpatterns = [
    url(r'^$', views.index, name='annotation_index'),
    url(r'^start_annotation/(?P<basicviz_experiment_id>\w+)/$',views.start_annotation, name='start_annotation'),
    url(r'^query/(?P<basicviz_experiment_id>\w+)/$', views.query_annotation, name='query_annotation'),
    url(r'^batch_query/(?P<db_name>\w+)/$', views.batch_query_annotation, name='batch_query_annotation'),
]