from django.conf.urls import include, url
from massbank import views

urlpatterns = [
    url(r'^generate_massbank/$', views.generate_massbank, name='generate_massbank'),
    url(r'^generate_massbank_multi_m2m/$', views.generate_massbank_multi_m2m,
        name='generate_massbank_multi_m2m'),
]
