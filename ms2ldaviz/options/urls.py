from django.conf.urls import include, url
from options import views

urlpatterns = [
    url(r'^view_experiment_options/(?P<experiment_id>\w+)/$', views.view_experiment_options,
        name='view_experiment_options'),
    url(r'^view_mf_experiment_options/(?P<mfe_id>\w+)/$', views.view_mf_experiment_options,
        name='view_mf_experiment_options'),
    url(r'^add_experiment_option/(?P<experiment_id>\w+)/$', views.add_experiment_option,
        name='add_experiment_option'),
    url(r'^delete_experiment_option/(?P<option_id>\w+)/$', views.delete_experiment_option,
        name='delete_experiment_option'),
    url(r'^edit_experiment_option/(?P<option_id>\w+)/$', views.edit_experiment_option,
        name='edit_experiment_option'),
    url(r'^add_mf_experiment_option/(?P<mfe_id>\w+)/$', views.add_mf_experiment_option,
        name='add_mf_experiment_option'),
    url(r'^delete_mf_experiment_option/(?P<option_id>\w+)/$', views.delete_mf_experiment_option,
        name='delete_mf_experiment_option'),
    url(r'^edit_mf_experiment_option/(?P<option_id>\w+)/$', views.edit_mf_experiment_option,
        name='edit_mf_experiment_option'),
]