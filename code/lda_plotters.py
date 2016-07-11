# -*- coding: utf-8 -*-

import plotly as plotly
from plotly.graph_objs import *


# Some useful plotting code (uses plotly)
# Should put this into a separate file
class VariationalLDAPlotter(object):
	def __init__(self,v_lda):
		plotly.offline.init_notebook_mode()
		self.v_lda = v_lda

	def bar_alpha(self):
		K = len(self.v_lda.alpha)
		data = []
		data.append(
			Bar(
				x = range(K),
				y = self.v_lda.alpha,
				)
			)
		plotly.offline.iplot({'data':data})
	def mean_gamma(self):
		K = len(self.v_lda.alpha)
		data = []
		data.append(
			Bar(
				x = range(K),
				y = self.v_lda.gamma_matrix.mean(axis=0),
				)
			)
		plotly.offline.iplot({'data':data})

class MultiFileVariationalLDAPlotter(object):
	def __init__(self,m_lda):
		plotly.offline.init_notebook_mode()
		self.m_lda = m_lda

	def multi_alpha(self,normalise=False,names=None):
		data = []
		K = self.m_lda.individual_lda[0].K
		for i,l in enumerate(self.m_lda.individual_lda):
			if normalise:
				a = l.alpha / l.alpha.sum()
			else:
				a = l.alpha
			if not names == None:
				name = names[i]
			else:
				name = 'trace {}'.format(i)
			data.append(
				Bar(
					x = range(K),
					y = a,
					name = name
					)
				)
		plotly.offline.iplot({'data':data})