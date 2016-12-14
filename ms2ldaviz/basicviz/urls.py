from django.conf.urls import include, url
from basicviz import views

# for single-file LDA experiments
lda_single_patterns = [
    url(r'^show_docs/(?P<experiment_id>\w+)/$', views.show_docs, name='show_docs'),
    url(r'^show_doc/(?P<doc_id>\w+)/$', views.show_doc, name='show_doc'),
    url(r'^start_viz/(?P<experiment_id>\w+)/$', views.start_viz, name='start_viz'),
    url(r'^start_annotated_viz/(?P<experiment_id>\w+)/$', views.start_annotated_viz,
        name='start_annotated_viz'),
    url(r'^get_graph/(?P<vo_id>\w+)/$', views.get_graph, name='get_graph'),
    # url(r'^get_annotated_graph/(?P<experiment_id>\w+)/$',views.get_annotated_graph,name='get_annotated_graph'),
    # url(r'^get_doc/(?P<doc_id>\w+)/$',views.get_doc,name='get_doc'),
    url(r'^get_doc_topics/(?P<doc_id>\w+)/$', views.get_doc_topics, name='get_doc_topics'),
    url(r'^view_parents/(?P<motif_id>\w+)/$', views.view_parents, name='get_parents'),
    url(r'^get_parents/(?P<motif_id>\w+)/(?P<vo_id>\w+)/$', views.get_parents, name='get_parents'),
    url(r'^get_parents/(?P<motif_id>\w+)/$', views.get_parents_no_vo, name='get_parents_no_vo'),
    url(r'^get_parents_metadata/(?P<motif_id>\w+)/$', views.get_parents_metadata, name='get_parents_metadata'),
    url(r'^get_all_parents_metadata/(?P<experiment_id>\w+)/$', views.get_all_parents_metadata, name='get_all_parents_metadata'),
    # url(r'^get_annotated_parents/(?P<motif_id>\w+)/$',views.get_annotated_parents,name='get_annotated_parents'),
    url(r'^view_mass2motifs/(?P<experiment_id>\w+)/$', views.view_mass2motifs,
        name='view_mass2motifs'),
    url(r'^topic_table/(?P<experiment_id>\w+)/$', views.topic_table, name='topic_table'),
    url(r'^document_pca/(?P<experiment_id>\w+)/$', views.document_pca, name='document_pca'),
    url(r'^topic_pca/(?P<experiment_id>\w+)/$', views.topic_pca, name='topic_pca'),
    url(r'^get_pca_data/(?P<experiment_id>\w+)/$', views.get_pca_data, name='get_pca_data'),
    url(r'^get_word_graph/(?P<motif_id>\w+)/(?P<vo_id>\w+)/$', views.get_word_graph,
        name='get_word_graph'),
    url(r'^get_intensity/(?P<motif_id>\w+)/(?P<vo_id>\w+)/$', views.get_intensity,
        name='get_intensity'),
    url(r'^view_word_graph/(?P<motif_id>\w+)/$', views.view_word_graph, name='view_word_graph'),
    url(r'^view_intensity/(?P<motif_id>\w+)/$', views.view_intensity, name='view_intensity'),
    url(r'^get_topic_pca_data/(?P<experiment_id>\w+)/$', views.get_topic_pca_data,
        name='get_topic_pca_data'),
    url(r'^rate_by_conserved_motif_rating/(?P<experiment_id>\w+)/$',
        views.rate_by_conserved_motif_rating, name='rate_by_conserved_motif_rating'),
    url(r'^validation/(?P<experiment_id>\w+)/$', views.validation, name='validation'),
    url(r'^toggle_dm2m/(?P<experiment_id>\w+)/(?P<dm2m_id>\w+)/$', views.toggle_dm2m,
        name='toggle_dm2m'),
    url(r'^dump_validations/(?P<experiment_id>\w+)/$', views.dump_validations,
        name='dump_validations'),
    url(r'^mass2motif_feature/(?P<fm2m_id>\w+)/$', views.mass2motif_feature,
        name='mass2motif_feature'),
    url(r'^extract_docs/(?P<experiment_id>\w+)/$', views.extract_docs, name='extract_docs'),
    url(r'^compute_topic_scores/(?P<experiment_id>\w+)/$', views.compute_topic_scores,
        name='compute_topic_scores'),
    url(r'^high_classyfire/(?P<experiment_id>\w+)/$', views.high_classyfire,
        name='high_classyfire'),    
]

# for multi-file LDA experiments
lda_multi_patterns = [
    url(r'^multi_alphas/(?P<mf_id>\w+)/$', views.multi_alphas, name='multi_alphas'),
    url(r'^alpha_pca/(?P<mf_id>\w+)/$', views.alpha_pca, name='alpha_pca'),
    url(r'^get_alpha_matrix/(?P<mf_id>\w+)/$', views.get_alpha_matrix, name='get_alpha_matrix'),
    url(r'^get_degree_matrix/(?P<mf_id>\w+)/$', views.get_degree_matrix, name='get_degree_matrix'),
    url(r'^view_multi_m2m/(?P<mf_id>\w+)/(?P<motif_name>\w+)/$', views.view_multi_m2m,
        name='view_multi_m2m'),
    url(r'^get_alphas/(?P<mf_id>\w+)/(?P<motif_name>\w+)/$', views.get_alphas, name='get_alphas'),
    url(r'^get_degrees/(?P<mf_id>\w+)/(?P<motif_name>\w+)/$', views.get_degrees, name='get_degrees'),
    url(r'^alpha_correlation/(?P<mf_id>\w+)/$', views.alpha_correlation, name='alpha_correlation'),
    url(r'^get_alpha_correlation_graph/(?P<acviz_id>\w+)/$', views.get_alpha_correlation_graph,
        name='alpha_correlation'),
    url(r'^dump_topic_molecules/(?P<m2m_id>\w+)/$', views.dump_topic_molecules,
        name='dump_topic_molecues'),
    url(r'^wipe_cache/(?P<mf_id>\w+)/$', views.wipe_cache, name='wipe_cache'),
    url(r'^get_doc_table/(?P<mf_id>\w+)/(?P<motif_name>\w+)/$', views.get_doc_table,
        name='get_doc_table'),
    url(r'^alpha_de/(?P<mfe_id>\w+)/$', views.alpha_de, name='alpha_de'),
    url(r'^get_individual_names/(?P<mf_id>\w+)/$', views.get_individual_names,
        name='get_individual_names'),
    url(r'^get_multifile_mass2motif_metadata/(?P<mf_id>\w+)/(?P<motif_name>\w+)/$',
        views.get_multifile_mass2motif_metadata, name='get_multifile_mass2motif_metadata'),
]

# for login
login_patterns = [
    url(r'^register/$', views.register, name='register'),
    url(r'^login/$', views.user_login, name='login'),
    url(r'^logout/$', views.user_logout, name='logout'),
]

# for experiment options
options_patterns = [
    url(r'^view_experiment_options/(?P<experiment_id>\w+)/$', views.view_experiment_options,
        name='view_experiment_options'),
    url(r'^view_mf_experiment_options/(?P<mfe_id>\w+)/$', views.view_mf_experiment_options,
        name='view_mf_experiment_options'),
    url(r'^add_experiment_option/(?P<experiment_id>\w+)/$', views.add_experiment_option,
        name='add_experiment_option'),
    url(r'^delete_experiment_option/(?P<option_id>\w+)/$', views.delete_experiment_option,
        name='delete_experiment_option'),
    url(r'^edit_experiment_option/(?P<option_id>\w+)/$', views.edit_experiment_option,
        name='edit_experiment_option'),
    url(r'^add_mf_experiment_option/(?P<mfe_id>\w+)/$', views.add_mf_experiment_option,
        name='add_mf_experiment_option'),
    url(r'^delete_mf_experiment_option/(?P<option_id>\w+)/$', views.delete_mf_experiment_option,
        name='delete_mf_experiment_option'),
    url(r'^edit_mf_experiment_option/(?P<option_id>\w+)/$', views.edit_mf_experiment_option,
        name='edit_mf_experiment_option'),
]

# for massbank stuff
massbank_patterns = [
    url(r'^generate_massbank/$', views.generate_massbank, name='generate_massbank'),
    url(r'^generate_massbank_multi_m2m/$', views.generate_massbank_multi_m2m,
        name='generate_massbank_multi_m2m'),
]

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^', include(lda_single_patterns)),
    url(r'^', include(lda_multi_patterns)),
    url(r'^', include(login_patterns)),
    url(r'^', include(options_patterns)),
    url(r'^', include(massbank_patterns)),
]
