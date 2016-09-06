from django.db import models
from django.contrib.auth.models import User
import jsonpickle

# Create your models here.


class Experiment(models.Model):
	name = models.CharField(max_length=128,unique=True)
	description = models.CharField(max_length=1024,null=True)
	status = models.CharField(max_length=128,null=True)
	def __unicode__(self):
		return self.name


class UserExperiment(models.Model):
	user = models.ForeignKey(User)
	experiment = models.ForeignKey(Experiment)


class Document(models.Model):
	name = models.CharField(max_length=32)
	experiment = models.ForeignKey(Experiment)
	metadata = models.CharField(max_length=2048,null=True)

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

	def get_mass(self):
		md = jsonpickle.decode(self.metadata)
		if 'parentmass' in md:
			return md['parentmass']
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

	logfc = property(get_logfc)
	mass = property(get_mass)
	csid = property(get_csid)
	inchikey = property(get_inchi)
	annotation = property(get_annotation)
	display_name = property(get_display_name)

	def __unicode__(self):
		return self.name

class Feature(models.Model):
	name = models.CharField(max_length=64)
	experiment = models.ForeignKey(Experiment)

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
	metadata = models.CharField(max_length=1024,null=True)

	def get_annotation(self):
		md = jsonpickle.decode(self.metadata)
		if 'annotation' in md:
			return md['annotation']
		else:
			return None

	annotation = property(get_annotation)

	def __unicode__(self):
		return self.name

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
	edge_thresh = models.FloatField(null = False)
	min_degree = models.IntegerField(null = False)
	just_annotated_docs = models.BooleanField(null = False)
	colour_by_logfc = models.BooleanField(null = False)
	discrete_colour = models.BooleanField(null = False)
	upper_colour_perc = models.IntegerField(null = False)
	lower_colour_perc = models.IntegerField(null = False)