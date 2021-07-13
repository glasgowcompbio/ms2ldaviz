import os
import sys

from registration.views import add_example_experiments

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ms2ldaviz.settings")

import django

django.setup()

from django.contrib.auth.models import User

if __name__ == '__main__':
    username = sys.argv[1]
    try:
        user = User.objects.get(username=username)
    except:
        print("No such user! {}".format(username))
        sys.exit(0)
    add_example_experiments(user)
