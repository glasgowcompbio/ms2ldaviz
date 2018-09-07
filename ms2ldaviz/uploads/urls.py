from django.conf.urls import include, url
from django.views.generic import TemplateView
from uploads import views

urlpatterns = [
    url(r'^create_experiment/$', views.create_experiment, name='create_experiment'),
    url(r'^process/$', views.process_experiment, name='process_experiment'),
    url(r'^ms1_format/$', TemplateView.as_view(template_name="uploads/ms1_format.html"), name='ms1_format'),
    url(r'^upload_experiment/$', views.upload_experiment, name='upload_experiment'),
]
