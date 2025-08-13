from django.conf import settings
from django.conf.urls import include, url  # For django versions before 2.0
from django.contrib import admin

from . import views

app_name='ms2ldaviz'

urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^people/', views.people, name='people'),
    url(r'^about/', views.about, name='about'),
    url(r'^user_guide/', views.user_guide, name='user_guide'),
    url(r'^disclaimer/', views.disclaimer, name='disclaimer'),
    url(r'^confidence/', views.confidence, name='confidence'),
    url(r'^grappelli/', include('grappelli.urls')),  # grappelli URLS
    url(r'^admin/', admin.site.urls),
    url(r'^motifdb/', include('motifdb.urls')),
]

# for django debug toolbar
if settings.DEBUG:
    import debug_toolbar
    urlpatterns = [
        url(r'^__debug__/', include(debug_toolbar.urls)),
    ] + urlpatterns
