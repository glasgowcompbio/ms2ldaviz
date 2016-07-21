from django.db import models
import jsonpickle

# Create your models here.
class Experiment(models.Model):
	name = models.CharField(max_length=128,unique=True)
	description = models.CharField(max_length=1024,null=True)
	def __unicode__(self):
		return self.name

class Document(models.Model):
	name = models.CharField(max_length=32)
	experiment = models.ForeignKey(Experiment)
	metadata = models.CharField(max_length=1024,null=True)

	def __unicode__(self):
		return self.name

class Feature(models.Model):
	name = models.CharField(max_length=32)
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
			return ""

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

	def __unicode__(self):
		return str(self.probability)

class FeatureMass2MotifInstance(models.Model):
	featureinstance = models.ForeignKey(FeatureInstance)
	mass2motif = models.ForeignKey(Mass2Motif)
	probability = models.FloatField()

	def __unicode__(self):
		return str(self.probability)
