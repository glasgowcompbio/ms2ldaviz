#!/bin/bash

ulimit -c unlimited
DJANGO_SETTINGS_MODULE='ms2ldaviz.settings' celery -A ms2ldaviz worker -l info