from django import forms
from django.contrib.auth.models import User

from basicviz.constants import AVAILABLE_OPTIONS
from basicviz.models import SystemOptions


class Mass2MotifMetadataForm(forms.Form):
    metadata = forms.CharField(max_length=256, required=False, label='Annotation', widget=forms.TextInput(attrs={'size': '80'}))
    short_annotation = forms.CharField(max_length=256, required=False, label='Short Annotation', widget=forms.TextInput(attrs={'size': '80'}))


class Mass2MotifMassbankForm(forms.Form):
    motif_id = forms.CharField(widget=forms.HiddenInput(attrs={'id': 'motif_id'}))
    mf_id = forms.CharField(widget=forms.HiddenInput(attrs={'id': 'mf_id'}))
    accession = forms.CharField(label='ACCESSION', required=True, widget=forms.TextInput(
        attrs={'id': 'accession', 'required': 'true', 'size': '50',
               'title': 'Identifier of the MassBank Record. Two or three letters site code, followed by record id.'})
    )
    authors = forms.CharField(label='AUTHORS', required=True, widget=forms.TextInput(
        attrs={'id': 'authors', 'required': 'true', 'size': '50',
               'title': 'Authors and Affiliations of MassBank Record.'})
    )
    comments = forms.CharField(label='COMMENTS', required=False, widget=forms.Textarea(
        attrs={'id': 'comments',
               'rows': '3', 'cols': '50',
               'title': 'In MassBank, COMMENT fields are often used to show the relations of the present record with other MassBank records and with data files.'})
    )
    ch_name = forms.CharField(label='CH$NAME', required=True, widget=forms.TextInput(
        attrs={'id': 'ch_name', 'required': 'true', 'size': '50', 'title': 'Name of the chemical compound analysed.'})
    )
    ch_compound_class = forms.CharField(label='CH$COMPOUND_CLASS', required=True, widget=forms.TextInput(
        attrs={'id': 'ch_compound_class', 'required': 'true', 'size': '50', 'title': 'Category of chemical compound.'})
    )
    ch_formula = forms.CharField(label='CH$FORMULA', required=True, widget=forms.TextInput(
        attrs={'id': 'ch_formula', 'required': 'true', 'size': '50',
               'title': 'Molecular formula of chemical compound.'})
    )
    ch_exact_mass = forms.CharField(label='CH$EXACT_MASS', required=True, widget=forms.TextInput(
        attrs={'id': 'ch_exact_mass', 'required': 'true', 'size': '12',
               'title': 'Monoisotopic mass of chemical compound.'})
    )
    ch_smiles = forms.CharField(label='CH$SMILES', required=True, widget=forms.TextInput(
        attrs={'id': 'ch_smiles', 'required': 'true', 'size': '50', 'title': 'SMILES string.'})
    )
    ch_iupac = forms.CharField(label='CH$IUPAC', required=True, widget=forms.TextInput(
        attrs={'id': 'ch_iupac', 'required': 'true', 'size': '50',
               'title': 'IUPAC International Chemical Identifier (InChI Code).'})
    )
    ch_link = forms.CharField(label='CH$LINK', required=False, widget=forms.Textarea(
        attrs={'id': 'ch_link',
               'rows': '3', 'cols': '50',
               'title': 'Identifier and link of chemical compound to external databases.'})
    )
    ac_instrument = forms.CharField(label='AC$INSTRUMENT', required=True, widget=forms.TextInput(
        attrs={'id': 'ac_instrument', 'required': 'true', 'size': '50',
               'title': 'Commercial Name and Model of (Chromatographic Separation Instrument, if any were coupled, and) Mass Spectrometer and Manufacturer.'})
    )
    ac_instrument_type = forms.CharField(label='AC$INSTRUMENT_TYPE', required=True, widget=forms.TextInput(
        attrs={'id': 'ac_instrument_type', 'required': 'true', 'size': '50', 'title': 'General Type of Instrument.'})
    )
    ac_mass_spectrometry_ion_mode = forms.CharField(label='AC$MASS_SPECTROMETRY: ION_MODE', required=True,
                                                    widget=forms.TextInput(
                                                        attrs={'id': 'ac_mass_spectrometry_ion_mode',
                                                               'required': 'true', 'size': '50'})
    )
    min_rel_int = forms.CharField(label='Min. relative intensity', required=True, widget=forms.TextInput(
        attrs={'id': 'min_rel_int', 'required': 'true', 'size': '5',
               'title': 'Minimum relative intensity to filter the features.'})
    )
