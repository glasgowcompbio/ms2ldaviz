from django.conf.urls import url
from registration import views

urlpatterns = [
    url(r'^register/$', views.register, name='register'),
    url(r'^login/$', views.user_login, name='login'),
    url(r'^logout/$', views.user_logout, name='logout'),
    url(r'^profile/', views.ProfileUpdate.as_view(), name='profile'),
]