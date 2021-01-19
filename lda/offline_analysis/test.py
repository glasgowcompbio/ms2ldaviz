#!/usr/bin/python


import sys
import getopt
import os
import requests

import os


def test_ms2lda():
    url = "http://ms2lda.org/"
    r = requests.get(url)
    r.raise_for_status()

    server_url = 'http://ms2lda.org/motifdb/'
    motifset_dict = requests.get(server_url+'list_motifsets/').json()

    db_list = []
    db_list.append(2)
    db_list.append(4)
    db_list.append(1)
    db_list.append(3)
    db_list.append(5)
    db_list.append(6)

    client = requests.session()
    token_output = client.get(server_url + 'initialise_api/').json()
    token = token_output['token']
    data = {'csrfmiddlewaretoken':token}
    data['motifset_id_list'] = db_list
    data['filter'] = 'True'

    output = client.post(server_url + 'get_motifset/',data = data).json()

test_ms2lda()