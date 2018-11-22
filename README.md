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

# Update lda subtree

The lda directory is a checkout of the https://github.com/sdrogers/lda repo, it can be synced using

```bash
git remote add lda https://github.com/sdrogers/lda.git
git subtree pull --prefix=lda lda master
```
