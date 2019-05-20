#!/bin/bash

ulimit -c unlimited
DJANGO_SETTINGS_MODULE='ms2ldaviz.settings_redisdebug' celery -A ms2ldaviz worker -l info --max-tasks-per-child 20 --concurrency 1
