# ms2ldaviz

A web application developed in Django+D3 to visualise how topics inferred from Latent Dirichlet Allocation can be used to assist in the unsupervised characterisation of fragmented (LC-MS-MS) metabolomics data.

Demo available at www.ms2lda.org (please email us to gain access)


# Run it

```
pipenv --python 2.7
pipenv install
pipenv shell
cd ms2ldaviz
```

In their own shell (within pipenv shell) run:
```
docker run --name some-redis -d -p 6379:6379 redis
docker run --name some-pg -d -p 5432:5432 -e POSTGRES_PASSWORD=j7z3rL40w9 -e POSTGRES_USER=django postgres
```
and
```
./start_celery_prod.sh
```
and
```
python manage.py migrate
python manage.py createsuperuser
python setup_feat.py
python manage.py runserver
```

Goto http://localhost:8000


# Run gensim lda

Requires server to be up and running.

Performs 3 steps:
1. Generate corpus/features from MS2 file
2. Run lda using gensim
3. Insert lda result into db


```bash
cd ms2ldaviz
./run_gensim.py corpus -f mgf myexp.mgf myexp.corpus.json
./run_gensim.py gensim myexp.corpus.json myexp.ldaresult.json
./run_gensim.py insert myexp.ldaresult.json stefanv myexp
```

## Run gensim with faster insert

This will exclude the lda info from the json file and write/import a gensim formatted lda dataset.

```bash
./run_gensim.py corpus -f mgf myexp.mgf myexp.corpus.json
./run_gensim.py gensim --ldaformat gensim myexp.corpus.json myexp.lda.gensim
./run_gensim.py insert_gensim myexp.corpus.json myexp.lda.gensim stefanv myexp
```

The last command inserts the gensim lda results into the database. 
This can also be done by using the web interface by going to `/uploads/upload_gensim_experiment/` url on the ms2lda server.
The gensim result must be tarballed with for example `tar -zcf myexp.lda.gensim.tar.gz myexp.lda.gensim*` and then uploaded in the form.

# Update lda subtree

The lda directory is a checkout of the https://github.com/sdrogers/lda repo, it can be synced using

```bash
git remote add lda https://github.com/sdrogers/lda.git
git subtree pull --prefix=lda lda master
```

# Docker

Run ms2lda website using docker-compose with

```bash
# Make sure lda/ is filled
docker-compose up -d
# For first time initialize db with
docker-compose run web python manage.py migrate
docker-compose run web python manage.py createsuperuser
docker-compose run web python setup_feat.py
```

Goto http://localhost:8000 to visit site

To run on different port then 8000 change `8000:8000` in `docker-compose.yml` file to `<insert port here>:8000`.

To clean up run
```bash
docker-compose down
```
