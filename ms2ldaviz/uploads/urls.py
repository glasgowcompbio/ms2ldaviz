from django.conf.urls import include, url
from uploads import views

urlpatterns = [
    url(r'^create_experiment/$', views.create_experiment, name='create_experiment'),
]
