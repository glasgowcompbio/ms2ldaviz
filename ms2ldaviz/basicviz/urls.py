from django.conf.urls import include, url
from basicviz.views.views_index import index
from basicviz.views.views_lda_admin import (
    list_log,
    show_log_file,
    list_all_experiments,
)
lda_admin_patterns = [
    url(r'^list_log/$', list_log,name = 'list_log'),
    url(r'^show_log_file/(?P<experiment_id>\w+)/$', show_log_file, name='show_log_file'),
    url(r'^list_all_experiments/$', list_all_experiments, name='list_all_experiments'),
]

urlpatterns = [
    url(r'^$', index, name='index'),
    url(r'^', include(lda_admin_patterns)),
]
