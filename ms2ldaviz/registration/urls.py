from django.conf.urls import url
from registration import views
import django.contrib.auth.views as auth_views

urlpatterns = [
    url(r'^register/$', views.register, name='register'),
    url(r'^login/$', views.user_login, name='login'),
    url(r'^logout/$', views.user_logout, name='logout'),
    url(r'^profile/$', views.ProfileUpdate.as_view(), name='profile'),
    url(r'^change_password/$', auth_views.password_change, {'post_change_redirect': 'profile'}, name='password_change'),
]