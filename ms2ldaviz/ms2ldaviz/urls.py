from django.conf.urls import patterns, include, url
from django.contrib import admin



urlpatterns = patterns('',
    # Examples:
    url(r'^$', 'ms2ldaviz.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^basicviz/',include('basicviz.urls')),
    url(r'^annotation/',include('annotation.urls')),
)
