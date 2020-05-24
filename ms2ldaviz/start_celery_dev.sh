#!/bin/bash

DJANGO_SETTINGS_MODULE='ms2ldaviz.settings_dev' celery -A ms2ldaviz worker -l info --max-tasks-per-child 20
