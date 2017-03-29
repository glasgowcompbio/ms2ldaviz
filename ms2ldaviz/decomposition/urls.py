# urls.py for decomposition
from django.conf.urls import include, url
from decomposition import views

urlpatterns = [
	url(r'^view_parents/(?P<decomposition_id>\w+)/(?P<mass2motif_id>\w+)/$', views.view_parents, name='view_parents_decomposition'),
	url(r'^get_parents/(?P<decomposition_id>\w+)/(?P<mass2motif_id>\w+)/$', views.get_parents, name='get_parents_decomposition'),
	url(r'^get_word_graph/(?P<mass2motif_id>\w+)/(?P<vo_id>\w+)/(?P<decomposition_id>\w+)/$', views.get_word_graph, name='get_word_graph'),
	url(r'^get_intensity_graph/(?P<mass2motif_id>\w+)/(?P<vo_id>\w+)/(?P<decomposition_id>\w+)/$', views.get_intensity_graph, name='get_intensity_graph'),
	url(r'^show_parents/(?P<decomposition_id>\w+)/$',views.show_parents,name='show_parents'),
	url(r'^show_doc/(?P<decomposition_id>\w+)/(?P<doc_id>\w+)/$',views.show_doc,name='show_doc'),
	url(r'^show_motifs/(?P<decomposition_id>\w+)/$',views.show_motifs,name='show_motifs'),
	url(r'^get_doc_topics/(?P<decomposition_id>\w+)/(?P<doc_id>\w+)/$',views.get_doc_topics,name='get_doc_topics'),
	url(r'^start_viz/(?P<decomposition_id>\w+)/$',views.start_viz,name='start_viz'),
	url(r'^get_graph/(?P<decomposition_id>\w+)/(?P<min_degree>\w+)/$',views.get_graph,name='get_graph'),
	url(r'^new_decomposition/(?P<experiment_id>\w+)/$',views.new_decomposition,name = 'new_decomposition'),
	url(r'^api/batch_decompose/$',views.batch_decompose,name = 'batch_decompose'),
	url(r'^api/batch_results/(?P<result_id>\w+)/$',views.batch_results,name = 'batch_results'),
	]
