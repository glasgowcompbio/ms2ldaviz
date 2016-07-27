from django.conf.urls import patterns, url
from basicviz import views

urlpatterns = patterns('',
        url(r'^$', views.index, name='index'),
		url(r'^show_docs/(?P<experiment_id>\w+)/$',views.show_docs,name='show_docs'),
		url(r'^show_doc/(?P<doc_id>\w+)/$',views.show_doc,name='show_doc'),
		url(r'^start_viz/(?P<experiment_id>\w+)/$',views.start_viz,name='start_viz'),
		url(r'^start_annotated_viz/(?P<experiment_id>\w+)/$',views.start_annotated_viz,name='start_annotated_viz'),
		url(r'^get_graph/(?P<experiment_id>\w+)/$',views.get_graph,name='get_graph'),
		url(r'^get_annotated_graph/(?P<experiment_id>\w+)/$',views.get_annotated_graph,name='get_annotated_graph'),
		# url(r'^get_doc/(?P<doc_id>\w+)/$',views.get_doc,name='get_doc'),
		url(r'^get_doc_topics/(?P<doc_id>\w+)/$',views.get_doc_topics,name='get_doc_topics'),
		url(r'^view_parents/(?P<motif_id>\w+)/$',views.view_parents,name='get_parents'),
		url(r'^get_parents/(?P<motif_id>\w+)/$',views.get_parents,name='get_parents'),
		url(r'^get_annotated_parents/(?P<motif_id>\w+)/$',views.get_annotated_parents,name='get_annotated_parents'),
		url(r'^view_mass2motifs/(?P<experiment_id>\w+)/$',views.view_mass2motifs,name='view_mass2motifs'),
		url(r'^document_pca/(?P<experiment_id>\w+)/$',views.document_pca,name='document_pca'),
		url(r'^get_pca_data/(?P<experiment_id>\w+)/$',views.get_pca_data,name='get_pca_data'),
		url(r'^get_features/(?P<motif_id>\w+)/$',views.get_features,name='get_features'),
        )