# ms2ldaviz

A web application developed in Django+D3 to visualise how topics inferred from Latent Dirichlet Allocation can be used to assist in the unsupervised characterisation of fragmented (LC-MS-MS) metabolomics data.

Demo available at www.ms2lda.org (please email us to gain access)


# Run it

```
pipenv --python 2.7
pipenv install
pipenv shell
cd ms2ldaviz
python manage.py migrate
python manage.py createsuperuser
python setup_feat.py
```

In their own shell (within pipenv shell) run:
```
docker run --name some-redis -d -p 6379:6379 redis
```
and
```
./start_celery_prod.sh
```
and
```
python manage.py runserver
```

Goto http://localhost:8000
