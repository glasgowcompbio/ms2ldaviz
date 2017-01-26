import datetime
import json

import jsonpickle
import numpy as np
import requests
from django.http import HttpResponse
from numpy import interp
from requests import RequestException

import basicviz.constants as constants
from basicviz.forms import Mass2MotifMassbankForm
from basicviz.models import Mass2Motif, Mass2MotifInstance, MultiFileExperiment, MultiLink

from splash import Spectrum, SpectrumType, Splash

def get_description(motif):
    exp_desc = motif.experiment.description
    if exp_desc is None:
        # look up in multi-file experiment
        links = MultiLink.objects.filter(experiment=motif.experiment)
        for link in links:
            mfe = link.multifileexperiment
            if mfe.description is not None:
                return mfe.description  # found a multi-file descrption
        return None  # found nothing
    else:
        return exp_desc  # found single-file experiment description


def get_massbank_form(motif, motif_features, mf_id=None):
    motif_id = motif.id

    # retrieve existing massbank dictionary for this motif or initialise a default one
    if motif.massbank_dict is not None:
        mb_dict = motif.massbank_dict
        is_new = False
    else:
        data = {'motif_id': motif_id}
        mb_dict = get_massbank_dict(data, motif, motif_features, 0)
        is_new = True

    print 'is_new', is_new
    print 'mb_dict', mb_dict

    # set to another form used when generating the massbank record
    massbank_form = Mass2MotifMassbankForm(initial={
        'motif_id': motif_id,
        'accession': mb_dict['accession'],
        'authors': mb_dict['authors'],
        'comments': '\n'.join(mb_dict['comments']),
        'ch_name': mb_dict['ch_name'],
        'ch_compound_class': mb_dict['ch_compound_class'],
        'ch_link': '\n'.join(mb_dict['ch_link']),
        'ac_instrument': mb_dict['ac_instrument'],
        'ac_instrument_type': mb_dict['ac_instrument_type'],
        'ac_mass_spectrometry_ion_mode': mb_dict['ac_mass_spectrometry_ion_mode'],
        'min_rel_int': 100 if is_new else mb_dict.get('min_rel_int', 100),
        'mf_id': mf_id if mf_id is not None else ''
    })
    return massbank_form

def get_massbank_dict(data, motif, motif_features, min_rel_int):
    default_accession = 'GP%06d' % int(data['motif_id'])
    accession = data.get('accession', default_accession)
    ms_type = 'MS2'

    if 'ac_mass_spectrometry_ion_mode' in data:
        ion_mode = data['ac_mass_spectrometry_ion_mode']
    else:
        # attempt to auto-detect from the experiment description
        exp_desc = get_description(motif)
        if exp_desc is not None:
            exp_desc = exp_desc.upper()
            ion_mode = 'POSITIVE' if 'POS' in exp_desc else 'NEGATIVE' if 'NEG' in exp_desc else 'Unknown'
        else:
            ion_mode = 'Unknown'

    # select the fragment/loss features to include
    peak_list = []
    for m2m in motif_features:
        tokens = m2m.feature.name.split('_')
        f_type = tokens[0]  # 'loss' or 'fragment'
        mz = float(tokens[1])
        if f_type == 'loss':  # represent neutral loss as negative m/z value
            mz = -mz
        abs_intensity = m2m.probability
        rel_intensity = m2m.probability
        row = (mz, abs_intensity, rel_intensity)
        peak_list.append(row)

    # this is [m/z, absolute intensity, relative intensity]
    peaks = np.array(peak_list)

    # sort by first (m/z) column
    mz = peaks[:, 0]
    peaks = peaks[mz.argsort()]

    # the probabilities * scale_fact are set to be the absolute intensities,
    # while the relative intensities are scaled from 1 ... 999 (from the manual)??
    scale_fact = 1000
    rel_range = [1, 999]
    abs_intensities = peaks[:, 1]
    min_prob = np.min(abs_intensities)
    max_prob = np.max(abs_intensities)
    rel_intensities = interp(abs_intensities, [min_prob, max_prob], rel_range)
    abs_intensities *= scale_fact  # do this only after computing the rel. intensities
    peaks[:, 2] = rel_intensities
    peaks[:, 1] = abs_intensities

    # filter features by the minimum relative intensity specified by the user
    pos = np.where(rel_intensities > min_rel_int)[0]
    peaks = peaks[pos, :]
    hash = get_splash(peaks)

    ch_name = motif.get_short_annotation()
    comments = data.get('comments', '').splitlines()
    ch_links = data.get('ch_link', '').splitlines()

    massbank_dict = {}
    massbank_dict['accession'] = accession
    massbank_dict['record_date'] = datetime.date.today().strftime('%Y.%m.%d')
    massbank_dict['authors'] = data.get('authors', constants.MASSBANK_AUTHORS)
    massbank_dict['license'] = constants.MASSBANK_LICENSE
    massbank_dict['copyright'] = constants.MASSBANK_COPYRIGHT
    massbank_dict['publication'] = constants.MASSBANK_PUBLICATION
    massbank_dict['ch_name'] = ch_name
    massbank_dict['ac_instrument'] = data.get('ac_instrument', constants.MASSBANK_AC_INSTRUMENT)
    massbank_dict['ac_instrument_type'] = data.get('ac_instrument_type', constants.MASSBANK_AC_INSTRUMENT_TYPE)
    massbank_dict['ms_type'] = ms_type
    massbank_dict['comments'] = comments
    massbank_dict['ch_link'] = ch_links
    massbank_dict['ac_mass_spectrometry_ion_mode'] = ion_mode
    massbank_dict['ac_ionisation'] = constants.MASSBANK_IONISATION
    massbank_dict['ms_data_processing'] = constants.MASSBANK_MS_DATA_PROCESSING
    massbank_dict['hash'] = hash
    massbank_dict['peaks'] = peaks

    tokens = [
        massbank_dict['ch_name'],
        massbank_dict['ac_instrument_type'],
        massbank_dict['ms_type']
    ]
    massbank_dict['record_title'] = ';'.join(tokens)
    massbank_dict['ch_compound_class'] = data.get('ch_compound_class', '')
    massbank_dict['ch_formula'] = 'NA'
    massbank_dict['ch_smiles'] = 'NA'
    massbank_dict['ch_iupac'] = 'NA'
    massbank_dict['ch_exact_mass'] = 'NA'
    massbank_dict['min_rel_int'] = min_rel_int

    return massbank_dict


def get_splash(peaks):

    peak_data = []
    for peak in peaks:
        row = (peak[0], peak[1]) # mz, intensity
        print row
        peak_data.append(row)

    spectrum = Spectrum(peak_data, SpectrumType.MS)
    hash = Splash().splash(spectrum)
    print hash
    return hash

def get_massbank_str(massbank_dict):
    print 'keys'
    for key in massbank_dict.keys():
        print '-', key

    output = []
    output.append('ACCESSION: %s' % massbank_dict['accession'])
    output.append('RECORD TITLE: %s' % massbank_dict['record_title'])
    output.append('DATE: %s' % massbank_dict['record_date'])
    output.append('AUTHORS: %s' % massbank_dict['authors'])
    output.append('LICENSE: %s' % massbank_dict['license'])
    output.append('COPYRIGHT: %s' % massbank_dict['copyright'])
    output.append('PUBLICATION: %s' % massbank_dict['publication'])
    for comment in massbank_dict['comments']:
        output.append('COMMENT: %s' % comment)
    output.append('CH$NAME: %s' % massbank_dict['ch_name'])
    output.append('CH$COMPOUND_CLASS: %s' % massbank_dict['ch_compound_class'])
    output.append('CH$FORMULA: %s' % massbank_dict['ch_formula'])
    output.append('CH$EXACT_MASS: %s' % massbank_dict['ch_exact_mass'])
    output.append('CH$SMILES: %s' % massbank_dict['ch_smiles'])
    output.append('CH$IUPAC: %s' % massbank_dict['ch_iupac'])
    for link in massbank_dict['ch_link']:
        output.append('CH$LINK: %s' % link)

    output.append('AC$INSTRUMENT: %s' % massbank_dict['ac_instrument'])
    output.append('AC$INSTRUMENT_TYPE: %s' % massbank_dict['ac_instrument_type'])
    output.append('AC$MASS_SPECTROMETRY: MS_TYPE %s' % massbank_dict['ms_type'])
    output.append('AC$MASS_SPECTROMETRY: ION_MODE %s' % massbank_dict['ac_mass_spectrometry_ion_mode'])
    output.append('AC$MASS_SPECTROMETRY: IONIZATION %s' % massbank_dict['ac_ionisation'])
    output.append('MS$DATA_PROCESSING: %s' % massbank_dict['ms_data_processing'])

    peaks = massbank_dict['peaks']
    output.append('PK$SPLASH: %s' % massbank_dict['hash'])
    output.append('PK$NUM_PEAK: %d' % len(peaks))
    output.append('PK$PEAK: m/z int. rel.int.')
    for peak in peaks:
        mz = '%.4f' % peak[0]
        abs_intensity = '%.4f' % peak[1]
        rel_intensity = '%d' % peak[2]
        output.append('%s %s %s' % (mz, abs_intensity, rel_intensity))

    output.append('//')
    output_str = '\n'.join(output)
    return output_str


def generate_massbank(request):
    if request.method == 'POST':

        # populate from post request
        data = {}
        keys = [
            'motif_id', 'accession', 'authors', 'comments',
            'ch_compound_class',
            'ch_smiles', 'ch_iupac', 'ch_link',
            'ac_instrument', 'ac_instrument_type', 'ac_mass_spectrometry_ion_mode',
            'min_rel_int'
        ]
        for key in keys:
            data[key] = request.POST.get(key)

        motif_id = data['motif_id']
        min_rel_int = int(data['min_rel_int'])

        # get the data in dictionary form
        motif = Mass2Motif.objects.get(id=motif_id)
        motif_features = Mass2MotifInstance.objects.filter(mass2motif=motif).order_by('-probability')
        mb_dict = get_massbank_dict(data, motif, motif_features, min_rel_int)

        # convert to string and add to the dictionary
        mb_string = get_massbank_str(mb_dict)
        del (mb_dict['peaks'])  # got error if we jsonpickle this numpy array .. ?
        mb_dict['massbank_record'] = mb_string

        # decode the metadata first, add the massbank field, then encode it back
        md = jsonpickle.decode(motif.metadata)
        md['massbank'] = mb_dict
        motif.metadata = jsonpickle.encode(md)
        motif.save()

        response_data = {}
        response_data['status'] = 'Massbank record has been generated. Please copy.'
        response_data['massbank_str'] = mb_string
        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )

    else:
        raise NotImplementedError


def generate_massbank_multi_m2m(request):
    if request.method == 'POST':

        # populate from post request
        data = {}
        keys = [
            'mf_id', 'motif_id', 'accession', 'authors', 'comments',
            'ch_compound_class',
            'ch_smiles', 'ch_iupac', 'ch_link',
            'ac_instrument', 'ac_instrument_type', 'ac_mass_spectrometry_ion_mode',
            'min_rel_int'
        ]
        for key in keys:
            data[key] = request.POST.get(key)

        mf_id = data['mf_id']
        first_motif_id = data['motif_id']
        min_rel_int = int(data['min_rel_int'])

        first_m2m = Mass2Motif.objects.get(id=first_motif_id)
        mfe = MultiFileExperiment.objects.get(id=mf_id)
        links = MultiLink.objects.filter(multifileexperiment=mfe).order_by('experiment__name')
        individuals = [l.experiment for l in links if l.experiment.status == 'all loaded']

        for individual in individuals:
            motif = Mass2Motif.objects.get(name=first_m2m.name, experiment=individual)

            # get the data in dictionary form
            motif_features = Mass2MotifInstance.objects.filter(mass2motif=motif).order_by('-probability')
            mb_dict = get_massbank_dict(data, motif, motif_features, min_rel_int)

            # convert to string and add to the dictionary
            mb_string = get_massbank_str(mb_dict)
            del (mb_dict['peaks'])  # got error if we jsonpickle this numpy array .. ?
            mb_dict['massbank_record'] = mb_string

            # decode the metadata first, add the massbank field, then encode it back
            md = jsonpickle.decode(motif.metadata)
            md['massbank'] = mb_dict
            motif.metadata = jsonpickle.encode(md)
            motif.save()

        response_data = {}
        response_data['status'] = 'Massbank record has been generated. Please copy.'
        response_data['massbank_str'] = mb_string
        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )

    else:
        raise NotImplementedError

