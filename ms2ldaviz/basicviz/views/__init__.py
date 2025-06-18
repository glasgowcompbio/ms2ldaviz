"""
basicviz.views package – trimmed for the lightweight site.

We publish only the modules that the URL-conf still needs.
Nothing here should import NumPy, SciPy, RDKit, scikit-learn, etc.
"""

from importlib import import_module

for _name in ("views_index", "views_lda_admin"):
    import_module(f"{__name__}.{_name}")

# from .views_index import *
# from .views_lda_single import *
# from .views_lda_multi import *
# from .views_lda_admin import *
