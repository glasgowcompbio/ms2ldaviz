import os
import jsonpickle
import datetime
from django.contrib.auth.models import User
from django.db import models
from django.conf import settings

from .constants import EXPERIMENT_STATUS_CODE,EXPERIMENT_TYPE, EXPERIMENT_DECOMPOSITION_SOURCE, EXPERIMENT_MS2_FORMAT


# Create your models here.
class MultiFileExperiment(models.Model):
    name = models.CharField(max_length=128, unique=True)
    description = models.CharField(max_length=1024, null=True)
    status = models.CharField(max_length=128, null=True)
    pca = models.TextField(null=True)
    alpha_matrix = models.TextField(null=True)
    degree_matrix = models.TextField(null=True)

    def __unicode__(self):
        return self.name


def get_upload_folder(instance, filename):
    upload_folder = "experiment_%s" % instance.pk
    return os.path.abspath(os.path.join(settings.MEDIA_ROOT, upload_folder, filename))


class BVFeatureSet(models.Model):
    name = models.CharField(max_length=64,unique = True)
    def __str__(self):
        bin_widths = {'binned_005':0.005,
                  'binned_01': 0.01,
                  'binned_05': 0.05,
                  'binned_1': 0.1,
                  'binned_5': 0.5}
        try:   
            return "{} Da".format(bin_widths[self.name])
        except:
            return self.name
    def __unicode__(self):
        return str(self)


class Experiment(models.Model):
    name = models.CharField(max_length=128, unique=True)
    description = models.CharField(max_length=1024, null=True)

    ready_code, _ = EXPERIMENT_STATUS_CODE[1]
    status = models.CharField(max_length=128, choices=EXPERIMENT_STATUS_CODE,
                              null=True, default=ready_code)

    ms2lda_code, _ = EXPERIMENT_TYPE[0]
    experiment_type = models.CharField(max_length=128, choices=EXPERIMENT_TYPE,
                              null=True, default=ms2lda_code)

    ms2lda_format_code, _ = EXPERIMENT_MS2_FORMAT[0]
    experiment_ms2_format = models.CharField(max_length=128, choices=EXPERIMENT_MS2_FORMAT,
                              null=True, default=ms2lda_format_code)

    ms2_file = models.FileField(blank=True, null=True, upload_to=get_upload_folder)
    csv_file = models.FileField(blank=True, null=True, upload_to=get_upload_folder)

    csv_mz_column = models.CharField(blank = True, null = True, max_length=128, default='mz')
    # csv_rt_column = models.CharField(blank = True, null = True, max_length=128)
    csv_rt_units = models.CharField(blank = True, null = True, choices = [('minutes','minutes'),('seconds','seconds')],max_length=128,default = 'seconds')

    csv_id_column = models.CharField(blank = True, null = True, max_length=128, default='scans')
    ms2_id_field = models.CharField(blank = True, null = True, max_length=128, default ='scans')
    

    # I don't think this should be stored here
    # as it precludes the same experiment being decomposed multiple times
    no, _ = EXPERIMENT_DECOMPOSITION_SOURCE[0]
    decomposition_source = models.CharField(max_length=1, choices=EXPERIMENT_DECOMPOSITION_SOURCE,
                              null=False, default=no)

    # perhaps these parameters should be moved to a separate table?
    isolation_window = models.FloatField(null=True, default=0.5)
    mz_tol = models.FloatField(null=True, default=5)
    rt_tol = models.FloatField(null=True, default=10) # seconds
    min_ms1_rt = models.FloatField(null=True, default=3*60) # seconds
    max_ms1_rt = models.FloatField(null=True, default=21*60) # seconds
    min_ms2_intensity = models.FloatField(null=True, default=5000.0)

    # Same with these -- this model is getting a bit bloated
    filter_duplicates = models.BooleanField(null = False,default = False)
    min_ms1_intensity = models.FloatField(null = True,default=0.0)
    duplicate_filter_mz_tol = models.FloatField(null = True,default = 0.5)
    duplicate_filter_rt_tol = models.FloatField(null = True,default = 16)

    n_its = models.IntegerField(null = True,default = 1000)
    K = models.IntegerField(null=True, default=300)
    featureset = models.ForeignKey(BVFeatureSet,null = True)


    def __unicode__(self):
        return self.name


    def save(self, *args, **kwargs):

        # we want to separate the uploaded files into folders by experiment id
        # but initially there's no id yet ...
        if self.pk is None:

            # temporarily unset the uploaded files
            uploaded_csv_file = self.csv_file
            uploaded_ms2_file = self.ms2_file
            self.csv_file = None
            self.ms2_file = None

            # save first to get the id
            super(Experiment, self).save(*args, **kwargs)

            # set the uploaded files back
            self.csv_file = uploaded_csv_file
            self.ms2_file = uploaded_ms2_file

        # actually does the saving here
        super(Experiment, self).save(*args, **kwargs)


class PublicExperiments(models.Model):
    experiment = models.ForeignKey(Experiment)


class MultiLink(models.Model):
    multifileexperiment = models.ForeignKey(MultiFileExperiment)
    experiment = models.ForeignKey(Experiment)


class ExtraUsers(models.Model):
    user = models.ForeignKey(User)


class UserExperiment(models.Model):
    user = models.ForeignKey(User)
    experiment = models.ForeignKey(Experiment)
    permission = models.CharField(max_length=24,null=False)


class Document(models.Model):
    name = models.CharField(max_length=64)
    experiment = models.ForeignKey(Experiment)
    metadata = models.CharField(max_length=2048, null=True)

    def get_annotation(self):
        md = jsonpickle.decode(self.metadata)
        if 'annotation' in md:
            return md['annotation']
        else:
            return None

    def get_inchi(self):
        md = jsonpickle.decode(self.metadata)
        if 'InChIKey' in md:
            return md['InChIKey']
        else:
            return None

    def get_csid(self):
        md = jsonpickle.decode(self.metadata)
        if 'csid' in md:
            return md['csid']
        else:
            return None

    def get_image_url(self):
        md = jsonpickle.decode(self.metadata)
        if 'csid' in md:
            # If this doc already has a csid, make the url
            return 'http://www.chemspider.com/ImagesHandler.ashx?id=' + str(self.csid)
        elif 'InChIKey' in md:
            # If it doesnt but it does have an InChIKey get the csid and make the image url
            from chemspipy import ChemSpider
            cs = ChemSpider('b07b7eb2-0ba7-40db-abc3-2a77a7544a3d')
            results = cs.search(md['InChIKey'])
            if results:
                # Return the image_url and also save the csid
                csid = results[0].csid
                md['csid'] = csid
                self.metadata = jsonpickle.encode(md)
                self.save()
                return results[0].image_url
            else:
                return None
        else:
            # If it has neither, no image!
            return None



    def get_mass(self):
        md = jsonpickle.decode(self.metadata)
        if 'parentmass' in md:
            return md['parentmass']
        elif 'mz' in md:
            return md['mz']
        else:
            return None

    def get_rt(self):
        md = jsonpickle.decode(self.metadata)
        if 'rt' in md:
            return md['rt']
        else:
            return None

    def get_display_name(self):
        display_name = self.name
        md = jsonpickle.decode(self.metadata)
        if 'common_name' in md:
            display_name = md['common_name']
        elif 'annotation' in md:
            display_name = md['annotation']
        return display_name

    def get_logfc(self):
        md = jsonpickle.decode(self.metadata)
        if 'logfc' in md:
            return md['logfc']
        else:
            return None

    def get_user_cols(self):
        md = jsonpickle.decode(self.metadata)
        if 'user_cols' in md:
            return md['user_cols']
        else:
            return None

    rt = property(get_rt)
    logfc = property(get_logfc)
    mass = property(get_mass)
    csid = property(get_csid)
    inchikey = property(get_inchi)
    annotation = property(get_annotation)
    display_name = property(get_display_name)
    image_url = property(get_image_url)
    user_cols = property(get_user_cols)

    def __unicode__(self):
        return self.name

class JobLog(models.Model):
    user = models.ForeignKey(User,null = True)
    experiment = models.ForeignKey(Experiment, null = True)
    timestamp = models.DateField(default= datetime.date.today, null = False)
    tasktype = models.CharField(max_length=1028, null = True)

class Feature(models.Model):
    name = models.CharField(max_length=64)
    experiment = models.ForeignKey(Experiment,null=True)
    min_mz = models.FloatField(null = True)
    max_mz = models.FloatField(null = True)
    featureset = models.ForeignKey(BVFeatureSet,null = True)

    def __unicode__(self):
        return self.name


class FeatureInstance(models.Model):
    document = models.ForeignKey(Document)
    feature = models.ForeignKey(Feature)
    intensity = models.FloatField()

    def __unicode__(self):
        return str(self.intensity)


class Mass2Motif(models.Model):
    name = models.CharField(max_length=32)
    experiment = models.ForeignKey(Experiment)
    metadata = models.CharField(max_length=1024 * 1024, null=True)

    linkmotif = models.ForeignKey('Mass2Motif',null = True)

    def get_annotation(self):
        md = jsonpickle.decode(self.metadata)
        if 'annotation' in md:
            return md['annotation']
        elif self.linkmotif:
            return self.linkmotif.annotation
        else:
            return None

    annotation = property(get_annotation)

    def get_short_annotation(self):
        md = jsonpickle.decode(self.metadata)
        if 'short_annotation' in md:
            return md['short_annotation']
        elif self.linkmotif:
            return self.linkmotif.short_annotation
        else:
            return self.annotation

    def get_massbank_dict(self):
        md = jsonpickle.decode(self.metadata)
        if 'massbank' in md:
            return md['massbank']
        else:
            return None


    
    massbank_dict = property(get_massbank_dict)
    short_annotation = property(get_short_annotation)

    def __unicode__(self):
        return self.name


class Alpha(models.Model):
    mass2motif = models.ForeignKey(Mass2Motif)
    value = models.FloatField()


class Mass2MotifInstance(models.Model):
    mass2motif = models.ForeignKey(Mass2Motif)
    feature = models.ForeignKey(Feature)
    probability = models.FloatField()

    def __unicode__(self):
        return str(self.probability)


class DocumentMass2Motif(models.Model):
    document = models.ForeignKey(Document)
    mass2motif = models.ForeignKey(Mass2Motif)
    probability = models.FloatField()
    validated = models.NullBooleanField()
    overlap_score = models.FloatField(null=True)

    def __unicode__(self):
        return str(self.probability)


class FeatureMass2MotifInstance(models.Model):
    featureinstance = models.ForeignKey(FeatureInstance)
    mass2motif = models.ForeignKey(Mass2Motif)
    probability = models.FloatField()

    def __unicode__(self):
        return str(self.probability)


class VizOptions(models.Model):
    experiment = models.ForeignKey(Experiment)
    # edge_thresh = models.FloatField(null=False)
    min_degree = models.IntegerField(null=False)
    # just_annotated_docs = models.BooleanField(null=False)
    # colour_by_logfc = models.BooleanField(null=False)
    # discrete_colour = models.BooleanField(null=False)
    # upper_colour_perc = models.IntegerField(null=False)
    # lower_colour_perc = models.IntegerField(null=False)
    # colour_topic_by_score = models.BooleanField(null=False)
    # random_seed = models.CharField(null=False, max_length=128)
    # edge_choice = models.CharField(null=False, max_length=128)
    ms1_analysis_id = models.IntegerField(null=True)


class AlphaCorrOptions(models.Model):
    multifileexperiment = models.ForeignKey(MultiFileExperiment)
    edge_thresh = models.FloatField(null=False)
    distance_score = models.CharField(null=False, max_length=24)
    normalise_alphas = models.BooleanField(null=False)
    max_edges = models.IntegerField(null=False)
    just_annotated = models.BooleanField(null=False)


class PeakSet(models.Model):
    multifileexperiment = models.ForeignKey(MultiFileExperiment)
    original_file = models.CharField(max_length=124, null=True)
    original_id = models.IntegerField(null=True)
    mz = models.FloatField(null=False)
    rt = models.FloatField(null=False)


class IntensityInstance(models.Model):
    peakset = models.ForeignKey(PeakSet)
    intensity = models.FloatField(null=True)
    experiment = models.ForeignKey(Experiment, null=True)
    document = models.ForeignKey(Document, null=True)


class SystemOptions(models.Model):
    key = models.CharField(null=False, max_length=124)
    value = models.CharField(null=False, max_length=124)
    experiment = models.ForeignKey(Experiment, null=True)

    class Meta:
        unique_together = ('key', 'experiment',)

    def __unicode__(self):
        return "{}  =  {}".format(self.key, self.value)

class MotifMatch(models.Model):
    frommotif = models.ForeignKey(Mass2Motif,related_name='frommotif')
    tomotif = models.ForeignKey(Mass2Motif,related_name='tomotif')
    score = models.FloatField(null = True)

    def __unicode__(self):
        return "{} <-> {} ({})".format(self.frommotif.name,self.tomotif.name,self.score)
