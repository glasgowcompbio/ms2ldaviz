# ms2ldaviz

A web application developed in Django+D3 to visualise how topics inferred from Latent Dirichlet Allocation can be used to assist in the unsupervised characterisation of fragmented (LC-MS-MS) metabolomics data.

Demo available at http://ms2lda.org

# Run it for development

```
pipenv --python 2.7
pipenv install
pipenv shell
cd ms2ldaviz
export DJANGO_SETTINGS_MODULE=ms2ldaviz.settings_redisdebug
```

In their own shell (within pipenv shell) run:
```
docker run --name some-redis -d -p 6379:6379 redis
docker run --name some-pg -d -p 5432:5432 -e POSTGRES_PASSWORD=j7z3rL40w9 -e POSTGRES_USER=django postgres
```
and
```
./start_celery_redisdebug.sh
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

Goto http://localhost:8001 to visit site

To run on different port then 8001 use `PORT=8123 docker-compose up -d`.

To clean up run
```bash
docker-compose down
```

# Environment Variables

The application uses the following environment variables for configuration:

- `DJANGO_SECRET_KEY`: The secret key used for cryptographic signing. If not set, a placeholder value is used (not secure for production).
- `DJANGO_DB_USER`: The database username. Defaults to 'postgres' if not set.
- `DJANGO_DB_PASSWORD`: The database password. If not set, a placeholder value is used (not secure for production).
- `ENABLE_ORIGINAL_JOB_SUBMISSION`: Controls whether users can create new experiments. Set to '0' to disable the Create Experiment button and redirect users away from experiment creation pages. Defaults to '1' (enabled) if not set.

For production deployments, it's essential to set these environment variables with secure values. You can set them in your environment before starting the application:

```bash
export DJANGO_SECRET_KEY="your_secure_secret_key"
export DJANGO_DB_USER="your_database_username"
export DJANGO_DB_PASSWORD="your_database_password"
export ENABLE_ORIGINAL_JOB_SUBMISSION="0"  # Set to 0 to disable new experiment creation
```

Or when using Docker, you can pass them in the docker-compose.yml file or as environment variables to the docker-compose command:

```bash
docker-compose run -e DJANGO_SECRET_KEY=your_secure_secret_key -e DJANGO_DB_USER=your_database_username -e DJANGO_DB_PASSWORD=your_database_password -e ENABLE_ORIGINAL_JOB_SUBMISSION=0 web python manage.py runserver
```
