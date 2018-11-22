from django.conf.urls import url
from annotation import views

urlpatterns = [
    url(r'^$', views.index, name='annotation_index'),
    url(r'^start_annotation/(?P<basicviz_experiment_id>\w+)/$',views.start_annotation, name='start_annotation'),
    url(r'^query/(?P<basicviz_experiment_id>\w+)/$', views.query_annotation, name='query_annotation'),
    url(r'^batch_query/(?P<db_name>\w+)/$', views.batch_query_annotation, name='batch_query_annotation'),
    url(r'^predict_substituents/(?P<experiment_id>\w+)/$',views.term_prediction,name='term_prediction'),
    url(r'^explore_terms/(?P<experiment_id>\w+)/$',views.explore_terms,name = 'explore_terms'),
    url(r'^list_docs_for_term/(?P<experiment_id>\w+)/(?P<term_id>\w+)/$',views.list_docs_for_term,name = 'list_docs_for_term'),
]