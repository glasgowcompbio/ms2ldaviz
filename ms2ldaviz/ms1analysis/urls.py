from django.conf.urls import include, url
from ms1analysis import views

urlpatterns = [
    url(r'^create_ms1analysis/(?P<experiment_id>\w+)/$', views.create_ms1analysis, name='create_ms1analysis'),
    # url(r'^process/$', views.process_experiment, name='process_experiment'),
    url(r'^create_ms1analysis_decomposition/(?P<decomposition_id>\w+)/$', views.create_ms1analysis_decomposition, name='create_ms1analysis_decomposition'),
]
