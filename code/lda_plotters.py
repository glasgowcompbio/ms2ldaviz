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



  # Colour by the top N topics in the document
  def plot_document_topic_colour(self,doc,precursor_mass = None,topn = 4,intensity_thresh = 5000,show_losses = False,title = None):
    eth = self.v_lda.get_expect_theta()
    pos = self.v_lda.doc_index[doc]
    tp = []
    # Find the highest probability topics
    for i,p in enumerate(eth[pos,:]):
        tp.append((i,p))
    tp = sorted(tp,key=lambda x:x[1],reverse=True)
    topics_to_plot = []
    for i in range(topn):
        topics_to_plot.append(tp[i][0])

    
    colours = [[255,0,0],[0,255,0],[0,0,255],[0,255,255]]
    data = []
    loss_opacity = 1.0
    top_colours = {}
    loss_colours = {}
    for i,t in enumerate(topics_to_plot):
        r = colours[i][0]
        g = colours[i][1]
        b = colours[i][2]
        top_colours[t] = ('rgb({},{},{})'.format(r,g,b))
        loss_colours[t] = ('rgba({},{},{},{})'.format(r,g,b,loss_opacity))    
    
    
    topics_plotted = [] # This will be appended to as we plot the first thing from each topic, for the legend
    max_intensity = 0.0
    for word in self.v_lda.corpus[doc]:
        if self.v_lda.corpus[doc][word] >= intensity_thresh:
            if word.startswith('fragment'):
                m = float(word.split('_')[1])
                intensity = self.v_lda.corpus[doc][word]
                if intensity >= max_intensity:
                    max_intensity = intensity
                cum = 0.0
                for t in topics_to_plot:
                    height = intensity*self.v_lda.phi_matrix[doc][word][t]
                    if t in topics_plotted:
                      s = Scatter(
                          x = [m,m],
                          y = [cum,cum+height],
                          mode = 'lines',
                          marker = dict(
                              color = top_colours[t]
                          ),
                          showlegend=False,
                      )
                    else:
                      name = "motif_{}".format(t)
                      s = Scatter(
                          x = [m,m],
                          y = [cum,cum+height],
                          mode = 'lines',
                          name = name,
                          marker = dict(
                              color = top_colours[t]
                          )
                      )
                      topics_plotted.append(t)
                    cum += height
                    data.append(s)
                s = Scatter(
                    x = [m,m],
                    y = [cum,intensity],
                    mode = 'lines',
                    marker = dict(
                        color = ('rgb(200,200,200)')
                    ),
                    showlegend=False,
                )
                data.append(s)
            if word.startswith('loss') and show_losses and not precursor_mass == None:
                loss_mass = float(word.split('_')[1])
                start = precursor_mass - loss_mass
                pos = start
                y = 0.9*self.v_lda.corpus[doc][word]
                for t in topics_to_plot:
                    width = loss_mass*self.v_lda.phi_matrix[doc][word][t]
                    if t in topics_plotted:
                        s = Scatter(
                            x = [pos,pos+width],
                            y = [y,y],
                            mode = 'lines',
                            marker = dict(
                                color = loss_colours[t]
                            ),
                            line = dict(
                                dash = 'dash'
                            ),
                            showlegend=False,
                        )
                    else:
                        name = "motif_{}".format(t)
                        s = Scatter(
                            x = [pos,pos+width],
                            y = [y,y],
                            mode = 'lines',
                            marker = dict(
                                color = loss_colours[t]
                            ),
                            line = dict(
                                dash = 'dash'
                            ),
                            showlegend=True,
                            name = name,
                        )
                        topics_plotted.append(t)
                    pos += width
                    data.append(s)
                s = Scatter(
                    x = [pos,precursor_mass],
                    y = [y,y],
                    mode = 'lines',
                    marker = dict(
                        color = ('rgba(200,200,200,{})'.format(loss_opacity))
                    ),
                    showlegend = False
                )
                data.append(s)
            
    if not precursor_mass == None:
      s = Scatter(
          x = [precursor_mass,precursor_mass],
          y = [0,max_intensity],
          mode = 'lines',
          marker = dict(
              color = ('rgb(255,0,0)')
          ),
          showlegend = False,
      )
      data.append(s)


    if title == None:
        title = str(doc)


    layout = Layout (
        showlegend=True,
        xaxis = dict(
            title = 'm/z',
        ),
        yaxis = dict(
            title = 'Intensity',
        ),
        title = title
    )
    plotly.offline.iplot({'data':data,'layout':layout})



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