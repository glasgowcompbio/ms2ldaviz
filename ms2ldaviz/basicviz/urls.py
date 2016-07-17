from django.conf.urls import patterns, url
from basicviz import views

urlpatterns = patterns('',
        url(r'^$', views.index, name='index'),
		url(r'^show_docs/(?P<experiment_id>\w+)/$',views.show_docs,name='show_docs'),
		url(r'^show_doc/(?P<doc_id>\w+)/$',views.show_doc,name='show_doc'),
        )