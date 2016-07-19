from django.conf.urls import patterns, url
from basicviz import views

urlpatterns = patterns('',
        url(r'^$', views.index, name='index'),
		url(r'^show_docs/(?P<experiment_id>\w+)/$',views.show_docs,name='show_docs'),
		url(r'^show_doc/(?P<doc_id>\w+)/$',views.show_doc,name='show_doc'),
		url(r'^start_viz/(?P<experiment_id>\w+)/$',views.start_viz,name='start_viz'),
		url(r'^get_graph/(?P<experiment_id>\w+)/$',views.get_graph,name='get_graph'),
		# url(r'^get_doc/(?P<doc_id>\w+)/$',views.get_doc,name='get_doc'),
		url(r'^get_doc_topics/(?P<doc_id>\w+)/$',views.get_doc_topics,name='get_doc_topics'),
		url(r'^view_parents/(?P<motif_id>\w+)/$',views.view_parents,name='get_parents'),
		url(r'^get_parents/(?P<motif_id>\w+)/$',views.get_parents,name='get_parents'),
		url(r'^view_mass2motifs/(?P<experiment_id>\w+)/$',views.view_mass2motifs,name='view_mass2motifs'),
        )