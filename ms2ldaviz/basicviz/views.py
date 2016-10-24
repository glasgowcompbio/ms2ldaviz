from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect,Http404

from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
import networkx as nx
from networkx.readwrite import json_graph
from sklearn.decomposition import PCA
import json
import jsonpickle
import csv
import numpy as np
from basicviz.forms import Mass2MotifMetadataForm,DocFilterForm,ValidationForm,VizForm,UserForm,TopicScoringForm,AlphaCorrelationForm,SystemOptionsForm,AlphaDEForm

from basicviz.models import Feature,Experiment,Document,FeatureInstance,DocumentMass2Motif,FeatureMass2MotifInstance,Mass2Motif,Mass2MotifInstance,VizOptions,UserExperiment,ExtraUsers,MultiFileExperiment,MultiLink,Alpha,AlphaCorrOptions,SystemOptions

from scipy.stats import pearsonr,ttest_ind

import math


available_options = [('doc_m2m_threshold','Probability threshold for showing document to mass2motif links'),
                     ('log_peakset_intensities','Whether or not to log the peakset intensities (true,false)'),
                     ('peakset_matching_tolerance','Tolerance to use when matching peaksets'),
                     ('heatmap_minimum_display_count','Minimum number of instances in a peakset to display it in the heatmap')]

@login_required(login_url='/basicviz/login/')
def index(request):
    userexperiments = UserExperiment.objects.filter(user = request.user)
    experiments = []
    for ue in userexperiments:
        experiments.append(ue.experiment)
    


    # Remove those that are multi ones
    exclude_individuals = []

    for experiment in experiments:
        links = MultiLink.objects.filter(experiment = experiment)
        if len(links) > 0:
            exclude_individuals += [l.experiment for l in links]


    print exclude_individuals
    for e in exclude_individuals:
        del experiments[experiments.index(e)]


    experiments = list(set(experiments))

    # experiments = Experiment.objects.all()
    context_dict = {'experiments':experiments}
    context_dict['user'] = request.user
    eu = ExtraUsers.objects.filter(user = request.user)

    mfe = MultiFileExperiment.objects.all()


    if len(eu) > 0:
        extra_user = True
    else:
        extra_user = False
    context_dict['extra_user'] = extra_user
    context_dict['mfe'] = mfe
    return render(request,'basicviz/basicviz.html',context_dict)

@login_required(login_url = '/basicviz/login/')
def user_logout(request):
    # Since we know the user is logged in, we can now just log them out.
    logout(request)

    # Take the user back to the homepage.
    return HttpResponseRedirect('/')


def topic_table(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    motifs = Mass2Motif.objects.filter(experiment = experiment)
    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['motifs'] = motifs
    return render(request,'basicviz/topic_table.html',context_dict)

def register(request):
    registered = False
    if request.method == 'POST':
        user_form = UserForm(data = request.POST)

        if user_form.is_valid():
            user = user_form.save()
            user.set_password(user.password)
            user.save()

            registered = True

        else:
            print user_form.errors

    else:
        user_form = UserForm()

    context_dict = {'user_form': user_form,'registered':registered}
    return render(request,
        'basicviz/register.html', context_dict)

# import time
def get_alpha_matrix(request,mf_id):
    if request.is_ajax():
        mfe = MultiFileExperiment.objects.get(id = mf_id)

        if not mfe.alpha_matrix:

            links = MultiLink.objects.filter(multifileexperiment = mfe)
            individuals = [l.experiment for l in links]
            motifs = Mass2Motif.objects.filter(experiment = individuals[0])
            

            # OLD CODE
            # t0 = time.time()
            # alp_vals = []
            # for motif in motifs:
            #     new_row = [motif.name,motif.annotation]
            #     tot = 0.0
            #     tot2 = 0.0
            #     for individual in individuals:
            #         motif_here = Mass2Motif.objects.get(name = motif.name,experiment = individual)
            #         # alp = Alpha.objects.get(mass2motif = motif_here)
            #         alp = motif_here.alpha_set.all()[0]
            #         new_row.append(alp.value)
            #         tot += alp.value
            #         tot2 += alp.value**2

            #     mu = tot/len(individuals)
            #     va = (tot2/len(individuals)) - mu**2
            #     new_row.append(va)
            #     alp_vals.append(new_row)

            # t1 = time.time()
            # print "TIME: {}".format(t1-t0)


            alp_vals = []
            for individual in individuals:
                motifs = individual.mass2motif_set.all().order_by('name')
                alp_vals.append([m.alpha_set.all()[0].value for m in motifs])

            alp_vals = map(list,zip(*alp_vals))
            alp_vals = [[motifs[i].name,motifs[i].annotation] + av + [float((np.array(av)/sum(av)).var())] for i,av in enumerate(alp_vals)]

            data = json.dumps(alp_vals)
            mfe.alpha_matrix = jsonpickle.encode(alp_vals)
            mfe.save()
        else:
            alp_vals = jsonpickle.decode(mfe.alpha_matrix)
            data = json.dumps(alp_vals)


        return HttpResponse(data,content_type = 'application/json')
    else:
        raise Http404

def get_degree_matrix(request,mf_id):
    if request.is_ajax():
        mfe = MultiFileExperiment.objects.get(id = mf_id)
        if not mfe.degree_matrix:
            links = MultiLink.objects.filter(multifileexperiment = mfe)
            individuals = [l.experiment for l in links]
            motifs = Mass2Motif.objects.filter(experiment = individuals[0])
            deg_vals = []
        
            # OLD CODE        
            # for motif in motifs:
            #     new_row = [motif.name,motif.annotation]
            #     for individual in individuals:
            #         motif_here = Mass2Motif.objects.get(name = motif.name,experiment = individual)
            #         docs = DocumentMass2Motif.objects.filter(mass2motif = motif_here)
            #         new_row.append(len(docs))

            #     deg_vals.append(new_row)

            
            for individual in individuals:

                doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = individual)
                if doc_m2m_threshold:
                    doc_m2m_threshold = float(doc_m2m_threshold)
                else:
                    doc_m2m_threshold = 0.0


                new_row = []
                motif_set = individual.mass2motif_set.all().order_by('name')
                for motif in motif_set:
                    dm2m = motif.documentmass2motif_set.all()
                    new_row.append(len([d for d in dm2m if d.probability > doc_m2m_threshold]))
                deg_vals.append(new_row)

            deg_vals = map(list,zip(*deg_vals))
            deg_vals = [[motif_set[i].name,motif_set[i].annotation]+dv for i,dv in enumerate(deg_vals)]

            data = json.dumps(deg_vals)
            mfe.degree_matrix = jsonpickle.encode(deg_vals)
            mfe.save()
        else:
            deg_vals = jsonpickle.decode(mfe.degree_matrix)
            data = json.dumps(deg_vals)
        return HttpResponse(data,content_type = 'application/json')
    else:
        raise Http404


def make_alpha_matrix(individuals,normalise = True):
    
    print "Creating alpha matrix"
    alp_vals = []
    for individual in individuals:
        motifs = individual.mass2motif_set.all().order_by('name')
        alp_vals.append([m.alpha_set.all()[0].value for m in motifs])

    alp_vals = map(list,zip(*alp_vals))
    new_alp_vals = []
    if normalise:
        for av in alp_vals:
            s = sum(av)
            nav = [a/s for a in av]
            new_alp_vals.append(nav)
        alp_vals = new_alp_vals

    # for motif in motifs:
    #     new_row = []
    #     deg_row = []
    #     tot = 0.0
    #     tot2 = 0.0
    #     for individual in individuals:
    #         thismotif = Mass2Motif.objects.get(name = motif.name,experiment = individual)
    #         alp = Alpha.objects.get(mass2motif = thismotif)
    #         new_row.append(alp.value)
    #         tot += alp.value
    #         tot2 += alp.value**2
    #         # docs = DocumentMass2Motif.objects.filter(mass2motif = thismotif)
    #         # deg_row.append(len(docs))
    #         deg_row.append(0)
    #     mu = tot / len(individuals)
    #     ss = (tot2)/len(individuals) - mu**2
    #     if normalise:
    #         new_row = [n/tot for n in new_row]
    #         tot = 1.0
    #         tot2 = sum([n**2 for n in new_row])
    #         mu = tot / len(individuals)
    #         ss = tot2/len(individuals) - mu**2
    #     if variances:
    #         new_row.append(ss)
    #     if add_motif:
    #         alp_vals.append((motif,new_row))
    #         degrees.append((motif,deg_row))
    #     else:
    #         alp_vals.append(new_row)
    #         degrees.append(deg_row)


    return alp_vals

def wipe_cache(request,mf_id):
    mfe = MultiFileExperiment.objects.get(id = mf_id)
    mfe.alpha_matrix = None
    mfe.degree_matrix = None
    mfe.save()
    return index(request)

def get_doc_table(request,mf_id,motif_name):
    mfe = MultiFileExperiment.objects.get(id = mf_id)
    links = MultiLink.objects.filter(multifileexperiment = mfe).order_by('experiment')
    individuals = [l.experiment for l in links]

    individual_motifs = {}
    for individual in individuals:
        thismotif = Mass2Motif.objects.get(experiment = individual,name = motif_name)
        individual_motifs[individual] = thismotif

    doc_table = []
    individual_names = []
    peaksets = {}
    peakset_list = []
    peakset_masses = []
    for i,individual in enumerate(individuals):
        doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = individual)
        if doc_m2m_threshold:
            doc_m2m_threshold = float(doc_m2m_threshold)
        else:
            doc_m2m_threshold = 0.0 # Default value
        individual_names.append(individual.name)
        docs = DocumentMass2Motif.objects.filter(mass2motif = individual_motifs[individual],probability__gte = doc_m2m_threshold)
        for doc in docs:
            peakset_index = -1
            ii = doc.document.intensityinstance_set.all()
            if len(ii) > 0:
                ii = ii[0]
                ps = ii.peakset
                if not ps in peaksets:
                    peaksets[ps] = {}
                    peakset_list.append(ps)
                    peakset_masses.append(ps.mz)
                peakset_index = peakset_list.index(ps)
                peaksets[ps][individual] = ii.intensity

            mz = 0
            rt = 0
            md = jsonpickle.decode(doc.document.metadata)
            if 'parentmass' in md:
                mz = md['parentmass']
            elif 'mz' in md:
                mz = md['mz']
            elif '_' in doc.document.name:
                split_name = doc.document.name.split('_')
                mz = float(split_name[0])
            if 'rt' in md:
                rt = md['rt']
            elif '_' in doc.document.name:
                split_name = doc.document.name.split('_')
                rt = float(split_name[1])

            
            doc_table.append([rt,mz,i,doc.probability,peakset_index])


    # Add the peaks to the peakset object that are not linked to a document (i.e. the MS1 peak is present, but it wasn't fragmented)
    for ps in peaksets:
        # Grab the intensity instances for this peakset
        intensity_instances = ps.intensityinstance_set.all()
        # Extract the individual experiments that are represented
        individuals_present = [i.experiment for i in intensity_instances]
        # Loop over the experiment
        for individual in individuals:
            # If the experiment is not in the current peakset but there is an intensity instance
            if (not individual in peaksets[ps]) and individual in individuals_present:
                # Find the intensity instance
                int_int = filter(lambda x : x.experiment == individual, intensity_instances)
                peaksets[ps][individual] = int_int[0].intensity
                print ps,individual,int_int[0].intensity

    intensity_table = []
    counts = []
    final_peaksets = []
    final_peakset_masses = []

    min_count_options = SystemOptions.objects.filter(key = 'heatmap_minimum_display_count')
    if len(min_count_options) > 0:
        min_count = int(min_count_options[0].value)
    else:
        min_count = 5

    log_peakset_intensities = True
    log_intensities_options = SystemOptions.objects.filter(key = 'log_peakset_intensities')
    if len(log_intensities_options) > 0:
        val = log_intensities_options[0].value
        if val == 'true':
            log_peakset_intensities = True
        else:
            log_peakset_intensities = False


    
    for peakset in peaksets:
        new_row = []
        for individual in individuals:
            new_row.append(peaksets[peakset].get(individual,0))
        count = sum([1 for i in new_row if i > 0])
        if min_count >= 0:
            nz_vals = [v for v in new_row if v > 0]
            if log_peakset_intensities:
                nz_vals = [np.log(v) for v in nz_vals]
                new_row = [np.log(v) if v > 0 else 0 for v in new_row]
            me = sum(nz_vals)/(1.0*len(nz_vals))
            va = sum([v**2 for v in nz_vals])/len(nz_vals) - me**2
            va = math.sqrt(va)
            if va > 0: # if variance is zero, skip...
                new_row_n = [(v - me)/va if v > 0 else 0 for v in new_row]
                intensity_table.append(new_row_n)
                counts.append(count)
                final_peaksets.append(peakset)


    # Order so that the most popular are at the top
    if len(final_peaksets) > 0:
        temp = zip(counts,intensity_table,final_peaksets)
        temp = sorted(temp,key = lambda x:x[0],reverse = True)
        counts,intensity_table,final_peaksets = zip(*temp)
        intensity_table = list(intensity_table)


    # Change the indexes in the doc table to match the new ordering
    for row in doc_table:
        old_ps_index = row[-1]
        if old_ps_index > -1:
            old_ps = peakset_list[old_ps_index]
            if old_ps in final_peaksets:
                new_ps_index = final_peaksets.index(old_ps)
            else:
                new_ps_index = -1
            row[-1] = new_ps_index
    

    final_peakset_masses = [p.mz for p in final_peaksets]
    final_peakset_rt = [p.rt for p in final_peaksets]




    return HttpResponse(json.dumps((individual_names,doc_table,intensity_table,final_peakset_masses,final_peakset_rt)),content_type = 'application/json')






def view_multi_m2m(request,mf_id,motif_name):
    mfe = MultiFileExperiment.objects.get(id = mf_id)
    links = MultiLink.objects.filter(multifileexperiment = mfe).order_by('experiment')
    individuals = [l.experiment for l in links if l.experiment.status == 'all loaded']
    context_dict = {'mfe':mfe}
    context_dict['motif_name'] = motif_name

    # get the features
    firstm2m = Mass2Motif.objects.get(name = motif_name,experiment = individuals[0])
    m2mfeatures = Mass2MotifInstance.objects.filter(mass2motif = firstm2m)
    m2mfeatures = sorted(m2mfeatures,key = lambda x: x.probability,reverse=True)
    context_dict['m2m_features'] = m2mfeatures


    individual_motifs = {}
    for individual in individuals:
        thism2m = Mass2Motif.objects.get(name = motif_name,experiment = individual)
        individual_motifs[individual] = thism2m


    context_dict['status'] = 'Edit metadata...'
    if request.method == 'POST':
        form = Mass2MotifMetadataForm(request.POST)
        if form.is_valid():
            new_annotation = form.cleaned_data['metadata']
            for individual in individual_motifs:
                motif = individual_motifs[individual]
                md = jsonpickle.decode(motif.metadata)
                if len(new_annotation) > 0:
                    md['annotation'] = new_annotation
                elif 'annotation' in md:
                    del md['annotation']
                motif.metadata = jsonpickle.encode(md)
                motif.save()
            context_dict['status'] = 'Metadata saved...'


    firstm2m = Mass2Motif.objects.get(name = motif_name,experiment = individuals[0])
    metadata_form = Mass2MotifMetadataForm(initial={'metadata':firstm2m.annotation})
    context_dict['metadata_form'] = metadata_form

    

    # Get the m2m in the individual models
    individual_m2m = []
    alps = []
    doc_table = []
    individual_names = []
    peaksets = {}
    peakset_list = []
    peakset_masses = []
    for i,individual in enumerate(individuals):
        alpha = Alpha.objects.get(mass2motif = individual_motifs[individual])
        doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = individual)
        if doc_m2m_threshold:
            doc_m2m_threshold = float(doc_m2m_threshold)
        else:
            doc_m2m_threshold = 0.0 # default
        docs = DocumentMass2Motif.objects.filter(mass2motif = individual_motifs[individual],probability__gte = doc_m2m_threshold)
        individual_m2m.append([individual,individual_motifs[individual],alpha,len(docs)])
        alps.append(alpha.value)
        
        # for doc in docs:
        #     peakset_index = -1
        #     ii = doc.document.intensityinstance_set.all()
        #     if len(ii) > 0:
        #         ii = ii[0]
        #         ps = ii.peakset
        #         if not ps in peaksets:
        #             peaksets[ps] = {}
        #             peakset_list.append(ps)
        #             peakset_masses.append(ps.mz)
        #         peakset_index = peakset_list.index(ps)
        #         peaksets[ps][individual] = ii.intensity

        #     mz = 0
        #     rt = 0
        #     md = jsonpickle.decode(doc.document.metadata)
        #     if 'parentmass' in md:
        #         mz = md['parentmass']
        #     elif 'mz' in md:
        #         mz = md['mz']
        #     elif '_' in doc.document.name:
        #         split_name = doc.document.name.split('_')
        #         mz = float(split_name[0])
        #     if 'rt' in md:
        #         rt = md['rt']
        #     elif '_' in doc.document.name:
        #         split_name = doc.document.name.split('_')
        #         rt = float(split_name[1])

            
        #     doc_table.append([rt,mz,i,doc.probability,peakset_index])
        # individual_names.append(individual.name)

    
    # intensity_table = []
    # counts = []
    # final_peaksets = []
    # final_peakset_masses = []

    # min_count_options = SystemOptions.objects.filter(key = 'heatmap_minimum_display_count')
    # if len(min_count_options) > 0:
    #     min_count = int(min_count_options[0].value)
    # else:
    #     min_count = 5

    # log_peakset_intensities = True
    # log_intensities_options = SystemOptions.objects.filter(key = 'log_peakset_intensities')
    # if len(log_intensities_options) > 0:
    #     val = log_intensities_options[0].value
    #     if val == 'true':
    #         log_peakset_intensities = True
    #     else:
    #         log_peakset_intensities = False

    # for peakset in peaksets:
    #     new_row = []
    #     for individual in individuals:
    #         new_row.append(peaksets[peakset].get(individual,0))
    #     count = sum([1 for i in new_row if i > 0])
    #     if min_count >= 5:
    #         nz_vals = [v for v in new_row if v > 0]
    #         if log_peakset_intensities:
    #             nz_vals = [np.log(v) for v in nz_vals]
    #         me = sum(nz_vals)/len(nz_vals)
    #         va = sum([v**2 for v in nz_vals])/len(nz_vals) - me**2
    #         va = math.sqrt(va)
    #         if va > 0: # if variance is zero, skip...
    #             new_row_n = [(v - me)/va if v > 0 else 0 for v in new_row]
    #             intensity_table.append(new_row_n)
    #             counts.append(count)
    #             final_peaksets.append(peakset)
    # for row in doc_table:
    #     old_ps_index = row[-1]
    #     if old_ps_index > -1:
    #         old_ps = peakset_list[old_ps_index]
    #         if old_ps in final_peaksets:
    #             new_ps_index = final_peaksets.index(old_ps)
    #         else:
    #             new_ps_index = -1
    #         row[-1] = new_ps_index
    

    # final_peakset_masses = [p.mz for p in final_peaksets]
    # final_peakset_rt = [p.rt for p in final_peaksets]

    # temp = zip(counts,intensity_table)
    # temp = sorted(temp,key = lambda x:x[0],reverse = True)
    # counts,intensity_table = zip(*temp)
    # intensity_table = list(intensity_table)

    # Compute the mean and variance
    tot_alps = sum(alps)
    m_alp = sum(alps)/len(alps)
    m_alp2 = sum([a**2 for a in alps])/len(alps)
    var = m_alp2 - m_alp**2
    context_dict['alpha_variance'] = var
    context_dict['alphas'] = zip([i.name for i in individuals],alps)
    context_dict['individual_m2m'] = individual_m2m
    # context_dict['doc_table'] = doc_table
    # context_dict['individual_names'] = json.dumps(individual_names)
    # context_dict['intensity_table'] = intensity_table
    # context_dict['peakset_masses'] = final_peakset_masses
    # context_dict['peakset_rt'] = final_peakset_rt


    return render(request,'basicviz/view_multi_m2m.html',context_dict)

def get_alphas(request,mf_id,motif_name):
    mfe = MultiFileExperiment.objects.get(id = mf_id)
    links = MultiLink.objects.filter(multifileexperiment = mfe).order_by('experiment')
    individuals = [l.experiment for l in links]
    alps = []
    for individual in individuals:
        m2m = Mass2Motif.objects.get(name = motif_name,experiment = individual)
        alpha = Alpha.objects.get(mass2motif = m2m)
        alps.append(alpha.value)
    
    alps = [[individuals[i].name,a] for i,a in enumerate(alps)]
    # alps = zip([i.name for i in individuals],alps)

    # alps = [0] + alps

    json_alps = json.dumps(alps)

    return HttpResponse(json_alps,content_type = 'application/json')

def get_degrees(request,mf_id,motif_name):
    mfe = MultiFileExperiment.objects.get(id = mf_id)
    links = MultiLink.objects.filter(multifileexperiment = mfe).order_by('experiment')
    individuals = [l.experiment for l in links]
    degs = []
    for individual in individuals:
        doc_m2m_threshold = get_option
        m2m = Mass2Motif.objects.get(name = motif_name,experiment = individual)
        doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = individual)
        if doc_m2m_threshold:
            doc_m2m_threshold = float(doc_m2m_threshold)
        else:
            doc_m2m_threshold = 0.0 # Default value

        docs = DocumentMass2Motif.objects.filter(mass2motif = m2m,probability__gte = doc_m2m_threshold)
        degs.append(len(docs))
    
    degs = zip([i.name for i in individuals],degs)

    # degs = [0] + degs

    json_degs = json.dumps(degs)

    return HttpResponse(json_degs,content_type = 'application/json')



def alpha_pca(request,mf_id):
    # Returns a json object to be rendered into a pca plot
    # PCA is pre-computed
    mfe = MultiFileExperiment.objects.get(id = mf_id)
    
    if mfe.pca:
        pca_data = jsonpickle.decode(mfe.pca)
    else:
        pca_data = []
    return HttpResponse(json.dumps(pca_data),content_type = 'application/json')



def multi_alphas(request,mf_id):
    mfe = MultiFileExperiment.objects.get(id = mf_id)
    context_dict = {'mfe':mfe}
    links = MultiLink.objects.filter(multifileexperiment = mfe)
    individuals = [l.experiment for l in links if l.experiment.status == 'all loaded']
    context_dict['individuals'] = individuals

    motifs = Mass2Motif.objects.filter(experiment = individuals[0])
    motif_names = [m.name for m in motifs]


    # alp_vals,degrees = make_alpha_matrix(motifs,individuals,normalise=False,variances=True,add_motif=True)
    alp_vals = []
    degrees = []
    context_dict['alp_vals'] = alp_vals
    context_dict['degrees'] = degrees
    context_dict['url'] = '/basicviz/alpha_pca/{}/'.format(mfe.id)
    return render(request,'basicviz/multi_alphas.html',context_dict)

def user_login(request):

    # If the request is a HTTP POST, try to pull out the relevant information.
    if request.method == 'POST':
        # Gather the username and password provided by the user.
        # This information is obtained from the login form.
                # We use request.POST.get('<variable>') as opposed to request.POST['<variable>'],
                # because the request.POST.get('<variable>') returns None, if the value does not exist,
                # while the request.POST['<variable>'] will raise key error exception
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Use Django's machinery to attempt to see if the username/password
        # combination is valid - a User object is returned if it is.
        user = authenticate(username=username, password=password)

        # If we have a User object, the details are correct.
        # If None (Python's way of representing the absence of a value), no user
        # with matching credentials was found.
        if user:
            # Is the account active? It could have been disabled.
            if user.is_active:
                # If the account is valid and active, we can log the user in.
                # We'll send the user back to the homepage.
                login(request, user)
                return HttpResponseRedirect('/basicviz/')
            else:
                # An inactive account was used - no logging in!
                return HttpResponse("Your account is disabled.")
        else:
            # Bad login details were provided. So we can't log the user in.
            print "Invalid login details: {0}, {1}".format(username, password)
            return HttpResponse("Invalid login details supplied.")

    # The request is not a HTTP POST, so display the login form.
    # This scenario would most likely be a HTTP GET.
    else:
        # No context variables to pass to the template system, hence the
        # blank dictionary object...
        return render(request, 'basicviz/login.html', {})


def alpha_correlation(request,mf_id):
    mfe = MultiFileExperiment.objects.get(id = mf_id)
    context_dict = {}
    context_dict['mfe'] = mfe

    if request.method == 'POST':
        form = AlphaCorrelationForm(request.POST)
        if form.is_valid():
            distance_score = form.cleaned_data['distance_score']
            edge_thresh = form.cleaned_data['edge_thresh']
            normalise_alphas = form.cleaned_data['normalise_alphas']
            max_edges = form.cleaned_data['max_edges']
            just_annotated = form.cleaned_data['just_annotated']
            acviz = AlphaCorrOptions.objects.get_or_create(multifileexperiment = mfe,
                                                            distance_score = distance_score,
                                                            edge_thresh = edge_thresh,
                                                            normalise_alphas = normalise_alphas,
                                                            max_edges = max_edges,
                                                            just_annotated = just_annotated)[0]
            context_dict['acviz'] = acviz
        else:
            context_dict['form'] = form
    else:
        context_dict['form'] = AlphaCorrelationForm()

    return render(request,'basicviz/alpha_correlation.html',context_dict)


def get_alpha_correlation_graph(request,acviz_id):
    from itertools import combinations
    acviz = AlphaCorrOptions.objects.get(id = acviz_id)
    mfe = acviz.multifileexperiment
    links = mfe.multilink_set.all()
    individuals = [l.experiment for l in links]
    an_experiment = links[0].experiment
    motifs = Mass2Motif.objects.filter(experiment = an_experiment).order_by('name')



    if mfe.alpha_matrix:
        alp_vals_with_names = jsonpickle.decode(mfe.alpha_matrix)
        alp_vals = []
        for av in alp_vals_with_names:
            newav = av[2:-1]
            alp_vals.append(newav)

    else:
        alp_vals = make_alpha_matrix(individuals,normalise = True)

    threshold = 0.98
    max_score = 0.0
    n_edges = 0
    motif_index = []
    an_motifs = []
    if acviz.just_annotated:
        for i,motif in enumerate(motifs):
            if 'annotation' in jsonpickle.decode(motif.metadata):
                an_motifs.append(motif)
                motif_index.append(i)
        motifs = an_motifs
    else:
        motif_index = range(len(motifs))
    # Add motifs as nodes
    G = nx.Graph()
    motif_names = []
    for motif in motifs:
        md = jsonpickle.decode(motif.metadata)
        display_name = md.get('annotation',motif.name)

        motif_names.append(motif.name)
        if 'annotation' in md:
            G.add_node(motif.name,name = display_name,col='#FF0000')
        else:
            G.add_node(motif.name,name = display_name,col='#333333')


        # (topic.name,group=2,name=name,
        #         size=topic_scale_factor * topics[topic],
        #         special = True, in_degree = topics[topic],
        #         score = 1,node_id = topic.id,is_topic = True,
        #         highlight_colour = highlight_colour)
    # add edges where the score is > thresh

    scores = []
    for i,j in combinations(range(len(motifs)),2):
        a1 = np.array(alp_vals[motif_index[i]])
        a2 = np.array(alp_vals[motif_index[j]])

        if acviz.normalise_alphas:
            a1n = a1/np.linalg.norm(a1)    
            a2n = a2/np.linalg.norm(a2)
        else:
            a1n = a1
            a2n = a2
        
        if acviz.distance_score == 'cosine':
            score = np.dot(a1n,a2n)
        elif acviz.distance_score == 'euclidean':
            score = np.sqrt((a1n-a2n)**2)
        elif acviz.distance_score == 'rms':
            score = np.sqrt(((a1n-a2n)**2).mean())
        elif acviz.distance_score == 'pearson':
            score = ((a1n - a1n.mean())*(a2n-a2n.mean())).mean()/(a1n.std()*a2n.std())
        

        scores.append((i,j,score))

    if acviz.distance_score == 'cosine' or acviz.distance_score == 'pearson':
        scores = sorted(scores,key = lambda x:x[2],reverse=True)
    else:
        scores = sorted(scores,key = lambda x:x[2])

    pos = 0
    while True:
        i,j,score = scores[pos]
        if (acviz.distance_score == 'cosine' or acviz.distance_score == 'pearson') and score > acviz.edge_thresh:
            G.add_edge(motif_names[i],motif_names[j],weight=score)
        elif (acviz.distance_score == 'euclidean' or acviz.distance_score == 'rms') and score > acviz.edge_thresh:
            G.add_edge(motif_names[i],motif_names[j])
        else:
            break
        pos += 1
        if pos > acviz.max_edges:
            break
        if pos >= len(scores):
            break

    d = json_graph.node_link_data(G)
    return HttpResponse(json.dumps(d),content_type = 'application/json')



def compute_topic_scores(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    context_dict = {'experiment':experiment}
    # Insert form logic here once form has been made
    # Start with a discrete hypergeo score
    documents = Document.objects.filter(experiment = experiment)
    md = jsonpickle.decode(documents[0].metadata)
    intensities = md['intensities']
    choices = [(i,i) for i in intensities.keys()]
    choices = sorted(choices,key = lambda x: x[0])

    if request.method == 'POST':
        form = TopicScoringForm(choices,request.POST)
        if form.is_valid():
            groups = form.cleaned_data['group1'] + form.cleaned_data['group2']
            
            intensities = []
            for document in documents:
                md = jsonpickle.decode(document.metadata)
                temp_intensity = []
                for group in groups:
                    temp_intensity.append(md['intensities'][group])
                intensities.append(temp_intensity)

            intensitynp = np.array(intensities)
            
            # Compute the logfc values
            logfc = []

            group1pos = range(len(form.cleaned_data['group1']))
            group2pos = range(len(form.cleaned_data['group1']),len(group1pos)+len(form.cleaned_data['group2']))


            SMALL = 1e-3
            for i,document in enumerate(documents):
                m1 = max(intensitynp[i,group1pos].mean(),SMALL)
                m2 = max(intensitynp[i,group2pos].mean(),SMALL)
                thisfc = np.log2(m1)-np.log2(m2)
                logfc.append(thisfc)
                if form.cleaned_data['storelogfc']:
                    md = jsonpickle.decode(document.metadata)
                    md['logfc'] = float(thisfc)
                    document.metadata = jsonpickle.encode(md)
                    document.save()

            logfc = np.array(logfc)


            # These should be set in a form
            lowperc = form.cleaned_data['lower_perc']
            upperc = form.cleaned_data['upper_perc']
            

            lfccopy = logfc.copy()
            lfccopy = np.sort(lfccopy)
            le = len(lfccopy)
            lowperc_value = lfccopy[int(np.floor(le*(lowperc/100.0)))]
            upperc_value = lfccopy[int(np.ceil(le*(upperc/100.0)))]

            total_above = len(np.where(logfc > upperc_value)[0])
            total_below = len(np.where(logfc < lowperc_value)[0])

            from scipy.stats.distributions import hypergeom

            M = len(documents)

            discrete_scores = []

            motifs = Mass2Motif.objects.filter(experiment = experiment)

            doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = experiment)
            if doc_m2m_threshold:
                doc_m2m_threshold = float(doc_m2m_threshold)
            else:
                doc_m2m_threshold = 0.00 # Default value



            n_done = 0
            for motif1ind,motif in enumerate(motifs):
                score_list = []
                doc_indices = []

                m2mdocs = DocumentMass2Motif.objects.filter(mass2motif = motif,probability__gte = doc_m2m_threshold)
                for m2mdoc in m2mdocs:
                    doc_indices.append(list(documents).index(m2mdoc.document))
                n_above = 0
                n_below = 0
                for ind in doc_indices:
                    if logfc[ind] < lowperc_value:
                        n_below += 1
                    if logfc[ind] > upperc_value:
                        n_above += 1
                n_sub_docs = len(m2mdocs)
                up_range = range(n_above,n_sub_docs+1)
                down_range = range(n_below,n_sub_docs+1)
                score_list.append(n_sub_docs)
                score_list.append(n_above)
                up_score = hypergeom.pmf(up_range,M,total_above,n_sub_docs).sum()
                score_list.append(up_score)
                score_list.append(n_below)
                down_score = hypergeom.pmf(down_range,M,total_below,n_sub_docs).sum()
                score_list.append(down_score)
                discrete_scores.append((motif,score_list))

                if form.cleaned_data['savetopicscores']:
                    md = jsonpickle.decode(motif.metadata)
                    md['upscore'] = float(up_score)
                    md['downscore'] = float(down_score)
                    motif.metadata = jsonpickle.encode(md)
                    motif.save()

                if form.cleaned_data['do_pairs']:
                    docs1 = set([m.document for m in m2mdocs])
                    for motif2 in motifs[motif1ind+1:]:
                        m2mdocs2 = DocumentMass2Motif.objects.filter(mass2motif = motif2,probability__gte = doc_m2m_threshold)
                        docs2 = set([m.document for m in m2mdocs2])
                        # Find the intersect
                        intersect = list(docs1 & docs2)
                        if len(intersect) > 0:
                            doc_indices = []
                            for doc in intersect:
                                doc_indices.append(list(documents).index(doc))
                            n_above = 0
                            n_below = 0
                            for ind in doc_indices:
                                if logfc[ind] < lowperc_value:
                                    n_below += 1
                                if logfc[ind] > upperc_value:
                                    n_above += 1
                            score_list = []
                            n_sub_docs = len(intersect)
                            up_range = range(n_above,n_sub_docs+1)
                            down_range = range(n_below,n_sub_docs+1)
                            score_list.append(n_sub_docs)
                            score_list.append(n_above)
                            up_score = hypergeom.pmf(up_range,M,total_above,n_sub_docs).sum()
                            score_list.append(up_score)
                            score_list.append(n_below)
                            down_score = hypergeom.pmf(down_range,M,total_below,n_sub_docs).sum()
                            score_list.append(down_score)
                            discrete_scores.append(("{}+{}".format(motif.name,motif2.name),score_list))
                n_done += 1
                print n_done


            context_dict['total_above'] = total_above
            context_dict['total_below'] = total_below
            context_dict['discrete_scores'] = discrete_scores
            context_dict['group1'] = form.cleaned_data['group1']
            context_dict['group2'] = form.cleaned_data['group2']

        else:
            # invalid form
            context_dict['topicscoringform'] = form
    else:
        form = TopicScoringForm(choices)
        context_dict['topicscoringform'] = form






    return render(request,'basicviz/compute_topic_scores.html',context_dict)




def show_docs(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    documents = Document.objects.filter(experiment = experiment)
    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['documents'] = documents
    context_dict['n_docs'] = len(documents)
    return render(request,'basicviz/show_docs.html',context_dict)

def show_doc(request,doc_id):
    document = Document.objects.get(id=doc_id)
    features = FeatureInstance.objects.filter(document = document)
    # features = sorted(features,key=lambda x:x.intensity,reverse=True)
    context_dict = {'document':document,'features':features}
    experiment = document.experiment
    context_dict['experiment'] = experiment
    doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = experiment)
    if doc_m2m_threshold:
        doc_m2m_threshold = float(doc_m2m_threshold)
    else:
        doc_m2m_threshold = 0.00 # Default value

    mass2motif_instances = DocumentMass2Motif.objects.filter(document = document,probability__gte = doc_m2m_threshold).order_by('-probability')
    context_dict['mass2motifs'] = mass2motif_instances
    feature_mass2motif_instances = []
    for feature in features:
        if feature.intensity > 0:
            feature_mass2motif_instances.append((feature,FeatureMass2MotifInstance.objects.filter(featureinstance=feature)))

    feature_mass2motif_instances = sorted(feature_mass2motif_instances,key = lambda x:x[0].intensity,reverse=True)

    if document.csid:
        context_dict['image_url'] = 'http://www.chemspider.com/ImagesHandler.ashx?id=' + str(document.csid)
        context_dict['csid'] = document.csid
    elif document.inchikey:
        from chemspipy import ChemSpider
        cs = ChemSpider('b07b7eb2-0ba7-40db-abc3-2a77a7544a3d')
        results = cs.search(document.inchikey)
        if results:
            context_dict['image_url'] = results[0].image_url
            context_dict['csid'] = results[0].csid

    context_dict['fm2m'] = feature_mass2motif_instances
    return render(request,'basicviz/show_doc.html',context_dict)

def view_parents(request,motif_id):
    motif = Mass2Motif.objects.get(id=motif_id)
    context_dict = {'mass2motif':motif}
    motif_features = Mass2MotifInstance.objects.filter(mass2motif = motif).order_by('-probability')
    total_prob = sum([m.probability for m in motif_features])
    context_dict['motif_features'] = motif_features
    context_dict['total_prob'] = total_prob

    doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = motif.experiment)
    if doc_m2m_threshold:
        doc_m2m_threshold = float(doc_m2m_threshold)
    else:
        doc_m2m_threshold = 0.00 # Default value
    dm2m = DocumentMass2Motif.objects.filter(mass2motif = motif,probability__gte = doc_m2m_threshold)
    context_dict['dm2ms'] = dm2m

    context_dict['status'] = 'Edit metadata...'
    if request.method == 'POST':
        form = Mass2MotifMetadataForm(request.POST)
        if form.is_valid():
            new_annotation = form.cleaned_data['metadata']
            md = jsonpickle.decode(motif.metadata)
            if len(new_annotation) > 0:
                md['annotation'] = new_annotation
            elif 'annotation' in md:
                del md['annotation']
            motif.metadata = jsonpickle.encode(md)
            motif.save()
            context_dict['status'] = 'Metadata saved...'



    metadata_form = Mass2MotifMetadataForm(initial={'metadata':motif.annotation})
    context_dict['metadata_form'] = metadata_form


    return render(request,'basicviz/view_parents.html',context_dict)


def mass2motif_feature(request,fm2m_id):
    mass2motif_feature = Mass2MotifInstance.objects.get(id = fm2m_id)
    context_dict = {}
    context_dict['mass2motif_feature'] = mass2motif_feature

    total_intensity = 0.0
    topic_intensity = 0.0
    n_docs = 0
    feature_instances = FeatureInstance.objects.filter(feature = mass2motif_feature.feature)
    docs = []
    for instance in feature_instances:
        total_intensity += instance.intensity
        fi_m2m = FeatureMass2MotifInstance.objects.filter(featureinstance = instance,mass2motif = mass2motif_feature.mass2motif)
        if len(fi_m2m) > 0:
            topic_intensity += fi_m2m[0].probability * instance.intensity
            if fi_m2m[0].probability >= 0.75:
                n_docs += 1
                docs.append(instance.document)

    context_dict['total_intensity'] = total_intensity
    context_dict['topic_intensity'] = topic_intensity
    context_dict['intensity_perc'] = 100.0*topic_intensity / total_intensity
    context_dict['n_docs'] = n_docs
    context_dict['docs'] = docs

    return render(request,'basicviz/mass2motif_feature.html',context_dict)

def get_parents(request,motif_id,vo_id):
    motif = Mass2Motif.objects.get(id=motif_id)
    vo = VizOptions.objects.all()
    viz_options  = VizOptions.objects.get(id = vo_id)
    docm2m = DocumentMass2Motif.objects.filter(mass2motif = motif,probability__gte=viz_options.edge_thresh).order_by('-probability')
    documents = [d.document for d in docm2m]
    parent_data = []
    for dm in docm2m:
        document = dm.document
        if viz_options.just_annotated_docs and document.annotation:
            parent_data.append(get_doc_for_plot(document.id,motif_id))
        elif not viz_options.just_annotated_docs:
            parent_data.append(get_doc_for_plot(document.id,motif_id))
    return HttpResponse(json.dumps(parent_data),content_type = 'application/json')


def get_parents_no_vo(request,motif_id):
    motif = Mass2Motif.objects.get(id=motif_id)

    doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = motif.experiment)
    if doc_m2m_threshold:
        doc_m2m_threshold = float(doc_m2m_threshold)
    else:
        doc_m2m_threshold = 0.00 # Default value


    docm2m = DocumentMass2Motif.objects.filter(mass2motif = motif,probability__gte = doc_m2m_threshold).order_by('-probability')
    documents = [d.document for d in docm2m]
    parent_data = []
    for dm in docm2m:
        document = dm.document
        parent_data.append(get_doc_for_plot(document.id,motif_id))
    return HttpResponse(json.dumps(parent_data),content_type = 'application/json')


# def get_annotated_parents(request,motif_id):
#     motif = Mass2Motif.objects.get(id=motif_id)
#     docm2m = DocumentMass2Motif.objects.filter(mass2motif = motif)
#     documents = [d.document for d in docm2m]
#     parent_data = []
#     for dm in docm2m:
#         if dm.probability > 0.05:
#             document = dm.document
#             if len(document.annotation) > 0:
#                 parent_data.append(get_doc_for_plot(document.id,motif_id))
#     return HttpResponse(json.dumps(parent_data),content_type = 'application/json')

def get_word_graph(request, motif_id, vo_id):

    motif = Mass2Motif.objects.get(id=motif_id)

    if not vo_id == 'nan':
        viz_options = VizOptions.objects.get(id = vo_id)
        edge_thresh = viz_options.edge_thresh
    else:
        edge_thresh = get_option('doc_m2m_threshold',experiment = motif.experiment)
        if edge_thresh:
            edge_thresh = float(edge_thresh)
        else:
            edge_thresh = 0.0


    m2mIns = Mass2MotifInstance.objects.filter(mass2motif = motif, probability__gte=0.01)
    m2mdocs = DocumentMass2Motif.objects.filter(mass2motif = motif, probability__gte=edge_thresh)
    colours = '#404080'
    features_data = {}
    for feat in m2mIns:                        
        features_data[feat.feature] = 0
    
    for doc in m2mdocs:
        feature_instances = FeatureInstance.objects.filter(document = doc.document)
        for ft in feature_instances:
            if ft.feature in features_data:
                features_data[ft.feature] += 1

    data_for_json = [] 
    data_for_json.append(len(m2mdocs))    
    sorted_feature_list = []

    for feature in features_data:
        if '.' in feature.name:
            split_name = feature.name.split('.')
            short_name = split_name[0]
            if len(split_name[1]) < 5:
                short_name += '.' + split_name[1]
            else:
                short_name += '.' + split_name[1][:5]
        else:
            short_name = feature.name
        sorted_feature_list.append([short_name,features_data[feature], colours]) 
    sorted_feature_list = sorted(sorted_feature_list,key =lambda x: x[1],reverse = True) 

    feature_list_full = []
    for feature in sorted_feature_list:
        feature_list_full.append(feature)
        #feature_list_full.append(["", 0, ""])

    data_for_json.append(feature_list_full)
    return HttpResponse(json.dumps(data_for_json), content_type = 'application/json')             

def view_word_graph(request, motif_id):
    motif = Mass2Motif.objects.get(id=motif_id)
    context_dict = {'mass2motif':motif}
    motif_features = Mass2MotifInstance.objects.filter(mass2motif = motif).order_by('-probability')
    context_dict['motif_features'] = motif_features
    return render(request,'basicviz/view_word_graph.html',context_dict)

def get_intensity(request, motif_id,vo_id):
    motif = Mass2Motif.objects.get(id=motif_id)
    features_m2m = Mass2MotifInstance.objects.filter(mass2motif = motif, probability__gte=0.01)
    features = [f.feature for f in features_m2m]
    colours = ['#404080', '#0080C0']
    total_intensity = {}
    mass2motif_intensity = {}

    #getting the total intensities of each feature
    for feature in features:
        feature_instances = FeatureInstance.objects.filter(feature = feature)
        total_intensity[feature] = 0.0
        mass2motif_intensity[feature] = 0.0
        for instance in feature_instances:
            total_intensity[feature] += instance.intensity
            fm2m = FeatureMass2MotifInstance.objects.filter(featureinstance = instance, mass2motif = motif)
            if len(fm2m) > 0:
                mass2motif_intensity[feature] += fm2m[0].probability * instance.intensity
    data_for_json = []
    features_list = []
    highest_intensity = 0;
    for feature in features:
        if mass2motif_intensity[feature] > 0:
            if '.' in feature.name:
                split_name = feature.name.split('.')
                short_name = split_name[0]
                if len(split_name[1]) < 5:
                    short_name += '.' + split_name[1]
                else:
                    short_name += '.' + split_name[1][:5]
            else:
                short_name = feature.name
            features_list.append((short_name,total_intensity[feature], colours[0]))
            features_list.append(('', mass2motif_intensity[feature], colours[1]))
            if total_intensity[feature] > highest_intensity:
                highest_intensity = total_intensity[feature]
            features_list.append(('', 0, ''))

    data_for_json.append(highest_intensity)
    data_for_json.append(features_list)
    return HttpResponse(json.dumps(data_for_json), content_type = 'application/json')

def view_intensity(request, motif_id):
    motif = Mass2Motif.objects.get(id=motif_id)
    context_dict = {'mass2motif':motif}
    motif_features = Mass2MotifInstance.objects.filter(mass2motif = motif).order_by('-probability')
    context_dict['motif_features'] = motif_features
    return render(request,'basicviz/view_intensity.html',context_dict)


def view_mass2motifs(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    mass2motifs = Mass2Motif.objects.filter(experiment = experiment).order_by('name')
    context_dict = {'mass2motifs':mass2motifs}
    context_dict['experiment'] = experiment
    return render(request,'basicviz/view_mass2motifs.html',context_dict)

def get_doc_for_plot(doc_id,motif_id = None,get_key = False):
    colours = ['red','green','black','yellow']
    document = Document.objects.get(id=doc_id)
    features = FeatureInstance.objects.filter(document = document)
    plot_fragments = []

    # Get the parent info
    metadata = jsonpickle.decode(document.metadata)
    if 'parentmass' in metadata:
        parent_mass = float(metadata['parentmass'])
    elif 'mz' in metadata:
        parent_mass = float(metadata['mz'])
    elif '_' in document.name:
        parent_mass = float(document.name.split('_')[0])
    else:
        parent_mass = 0.0
    probability = "na"

    if not motif_id == None:
        m2m = Mass2Motif.objects.get(id = motif_id)
        dm2m = DocumentMass2Motif.objects.get(mass2motif = m2m,document = document)
        probability = dm2m.probability

    parent_data = (parent_mass,100.0,document.display_name,document.annotation,probability)
    plot_fragments.append(parent_data)
    child_data = []

    # Only colours the first five
    if motif_id == None:
        topic_colours = {}
        topics = sorted(DocumentMass2Motif.objects.filter(document=document),key=lambda x:x.probability,reverse=True)
        topics_to_plot = []
        for i in range(4):
            if i == len(topics):
                break
            topics_to_plot.append(topics[i].mass2motif)
            topic_colours[topics[i].mass2motif] = colours[i]
    else:
        topic = Mass2Motif.objects.get(id = motif_id)
        topics_to_plot = [topic]
        topic_colours = {topic:'red'}

    max_intensity = 0.0
    for feature_instance in features:
        if feature_instance.intensity > max_intensity:
            max_intensity = feature_instance.intensity

    if len(features) > 0:
        for feature_instance in features:
            phi_values = FeatureMass2MotifInstance.objects.filter(featureinstance = feature_instance)
            mass = float(feature_instance.feature.name.split('_')[1])
            this_intensity = feature_instance.intensity*100.0/max_intensity
            feature_name = feature_instance.feature.name
            if feature_name.startswith('fragment'):
                cum_pos = 0.0
                other_topics = 0.0
                for phi_value in phi_values:
                    if phi_value.mass2motif in topics_to_plot:
                        proportion = phi_value.probability*this_intensity
                        colour = topic_colours[phi_value.mass2motif]
                        child_data.append((mass,mass,cum_pos,cum_pos + proportion,1,colour,feature_name))
                        cum_pos += proportion
                    else:
                        proportion = phi_value.probability*this_intensity
                        other_topics += proportion
                child_data.append((mass,mass,this_intensity - other_topics,this_intensity,1,'gray',feature_name))
            else:
                cum_pos = parent_mass - mass
                other_topics = 0.0
                for phi_value in phi_values:
                    if phi_value.mass2motif in topics_to_plot:
                        proportion = mass * phi_value.probability
                        colour = topic_colours[phi_value.mass2motif]
                        child_data.append((cum_pos,cum_pos+proportion,this_intensity,this_intensity,0,colour,feature_name))
                        cum_pos += proportion
                    else:
                        proportion = mass * phi_value.probability
                        other_topics += proportion
                child_data.append((parent_mass - other_topics,parent_mass,this_intensity,this_intensity,0,'gray',feature_name))
    plot_fragments.append(child_data)

    if get_key:
        key = []
        for topic in topic_colours:
            key.append((topic.name,topic_colours[topic]))
        return [plot_fragments],key

    return plot_fragments


def get_doc_topics(request,doc_id):
    plot_fragments = [get_doc_for_plot(doc_id,get_key = True)]
    return HttpResponse(json.dumps(plot_fragments),content_type='application/json')


def start_viz(request,experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    context_dict = {'experiment':experiment}

    if request.method == 'POST':
        viz_form = VizForm(request.POST)
        if viz_form.is_valid():
            min_degree = viz_form.cleaned_data['min_degree']
            edge_thresh = viz_form.cleaned_data['edge_thresh']
            j_a_n = viz_form.cleaned_data['just_annotated_docs']
            colour_by_logfc = viz_form.cleaned_data['colour_by_logfc']
            discrete_colour = viz_form.cleaned_data['discrete_colour']
            lower_colour_perc = viz_form.cleaned_data['lower_colour_perc']
            upper_colour_perc = viz_form.cleaned_data['upper_colour_perc']
            colour_topic_by_score = viz_form.cleaned_data['colour_topic_by_score']
            random_seed = viz_form.cleaned_data['random_seed']
            edge_choice = viz_form.cleaned_data['edge_choice']
            edge_choice = edge_choice[0].encode('ascii', 'ignore') # should turn the unicode into ascii
            vo = VizOptions.objects.get_or_create(experiment = experiment, 
                                                  min_degree = min_degree, 
                                                  edge_thresh = edge_thresh,
                                                  just_annotated_docs = j_a_n,
                                                  colour_by_logfc = colour_by_logfc,
                                                  discrete_colour = discrete_colour,
                                                  lower_colour_perc = lower_colour_perc,
                                                  upper_colour_perc = upper_colour_perc,
                                                  colour_topic_by_score = colour_topic_by_score,
                                                  random_seed = random_seed,
                                                  edge_choice = edge_choice)[0]
            context_dict['viz_options'] = vo

        else:
            context_dict['viz_form'] = viz_form
    else:
        viz_form = VizForm()
        context_dict['viz_form'] = viz_form

    if 'viz_form' in context_dict:
        return render(request,'basicviz/viz_form.html',context_dict)
    else:
        initial_motif = Mass2Motif.objects.filter(experiment = experiment)[0]
        context_dict['initial_motif'] = initial_motif
        return render(request,'basicviz/graph.html',context_dict)

def start_annotated_viz(request,experiment_id):
    # Is this function ever called??
    experiment = Experiment.objects.get(id=experiment_id)
    context_dict = {'experiment':experiment}
    # This is a bit of a hack to make sure that the initial motif is one in the graph
    documents = Document.objects.filter(experiment = experiment)
    for document in documents:
        if len(document.annotation) > 0:
            for docm2m in DocumentMass2Motif.objects.filter(document = document):
                if docm2m.probability > 0.05:
                    context_dict['initial_motif'] = docm2m.mass2motif
                    break
    return render(request,'basicviz/annotated_graph.html',context_dict)


def get_graph(request,vo_id):
    viz_options = VizOptions.objects.get(id = vo_id)
    experiment = viz_options.experiment
    G = make_graph(experiment,min_degree = viz_options.min_degree,
                                        edge_thresh = viz_options.edge_thresh,
                                        just_annotated_docs = viz_options.just_annotated_docs,
                                        colour_by_logfc = viz_options.colour_by_logfc,
                                        discrete_colour = viz_options.discrete_colour,
                                        lower_colour_perc = viz_options.lower_colour_perc,
                                        upper_colour_perc = viz_options.upper_colour_perc,
                                        colour_topic_by_score = viz_options.colour_topic_by_score,
                                        edge_choice = viz_options.edge_choice)
    d = json_graph.node_link_data(G)
    return HttpResponse(json.dumps(d),content_type='application/json')



def make_graph(experiment,edge_thresh = 0.05,min_degree = 5,
    topic_scale_factor = 5,edge_scale_factor=5,just_annotated_docs = False,
    colour_by_logfc = False,discrete_colour = False,lower_colour_perc = 10,upper_colour_perc = 90,
    colour_topic_by_score = False,edge_choice = 'probability'):
    mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
    # Find the degrees
    topics = {}
    for mass2motif in mass2motifs:
        topics[mass2motif] = 0
        if edge_choice == 'probability':
            docm2ms = DocumentMass2Motif.objects.filter(mass2motif=mass2motif,probability__gte=edge_thresh)
        else:
            docm2ms = DocumentMass2Motif.objects.filter(mass2motif=mass2motif,overlap_score__gte=edge_thresh)
        for d in docm2ms:
            if just_annotated_docs and d.document.annotation:
                topics[mass2motif] += 1
            elif (not just_annotated_docs):
                topics[mass2motif] += 1
    to_remove = []
    for topic in topics:
        if topics[topic] < min_degree:
            to_remove.append(topic)
    for topic in to_remove:
        del topics[topic]

    print "First"
    # Add the topics to the graph
    G = nx.Graph()
    for topic in topics:
        metadata = jsonpickle.decode(topic.metadata)
        if colour_topic_by_score:
            upscore = metadata.get('upscore',1.0)
            downscore = metadata.get('downscore',1.0)
            if upscore < 0.05:
                highlight_colour = '#0000FF'
            elif downscore < 0.05:
                highlight_colour = '#FF0000'
            else:
                highlight_colour = '#AAAAAA'
            name = metadata.get('annotation',topic.name)
            G.add_node(topic.name,group=2,name=name,
                size=topic_scale_factor * topics[topic],
                special = True, in_degree = topics[topic],
                score = 1,node_id = topic.id,is_topic = True,
                highlight_colour = highlight_colour)

        else:
            if 'annotation' in metadata:
                G.add_node(topic.name,group=2,name=metadata['annotation'],
                    size=topic_scale_factor * topics[topic],
                    special = True, in_degree = topics[topic],
                    score = 1,node_id = topic.id,is_topic = True)
            else:
                G.add_node(topic.name,group=2,name=topic.name,
                    size=topic_scale_factor * topics[topic],
                    special = False, in_degree = topics[topic],
                    score = 1,node_id = topic.id,is_topic = True)

    documents = Document.objects.filter(experiment = experiment)
    if colour_by_logfc:
        all_logfc_vals = []
        if colour_by_logfc:
            for document in documents:
                if document.logfc:
                    val = float(document.logfc)
                    if not np.abs(val) == np.inf:
                        all_logfc_vals.append(float(document.logfc))
        logfc_vals = np.sort(np.array(all_logfc_vals))


        perc_lower = logfc_vals[int(np.floor((lower_colour_perc/100.0)*len(logfc_vals)))]
        perc_upper = logfc_vals[int(np.ceil((upper_colour_perc/100.0)*len(logfc_vals)))]


        lowcol = [255,0,0]
        endcol = [0,0,255]




    if just_annotated_docs:
        new_documents = []
        for document in documents:
            if document.annotation:
                new_documents.append(document)
        
        documents = new_documents

    doc_nodes = []

    print "Second"

    for docm2m in DocumentMass2Motif.objects.filter(document__in=documents,mass2motif__in=topics,probability__gte=edge_thresh):
            # if docm2m.mass2motif in topics:
        if not docm2m.document in doc_nodes:
            metadata = jsonpickle.decode(docm2m.document.metadata)
            if 'compound' in metadata:
              name = metadata['compound']
            elif 'annotation' in metadata:
              name = metadata['annotation']  
            else:
              name = docm2m.document.name
            if not colour_by_logfc:
                G.add_node(docm2m.document.name,group=1,name = name,size=20,
                        type='square',peakid = docm2m.document.name,special=False,
                        in_degree=0,score=0,is_topic = False)
            else:
                if docm2m.document.logfc:
                    lfc = float(docm2m.document.logfc)
                    if lfc > perc_upper or lfc == np.inf:
                        col = "#{}{}{}".format('00','00','FF')
                    elif lfc < perc_lower or -lfc == np.inf:
                        col = "#{}{}{}".format('FF','00','00')
                    else:
                        if not discrete_colour:
                            pos = (lfc - perc_lower)/(perc_upper-perc_lower)
                            r = lowcol[0] + int(pos*(endcol[0] - lowcol[0]))
                            g = lowcol[1] + int(pos*(endcol[1] - lowcol[1]))
                            b = lowcol[2] + int(pos*(endcol[2] - lowcol[2]))
                            col = "#{}{}{}".format("{:02x}".format(r),"{:02x}".format(g),"{:02x}".format(b))
                        else:
                            col = '#FFFFFF'
                else:
                    col = '#FFFFFF'
                G.add_node(docm2m.document.name,group=1,name = name,size=20,
                        type='square',peakid = docm2m.document.name,special=True,
                        highlight_colour = col,logfc = docm2m.document.logfc,
                        in_degree=0,score=0,is_topic = False)

            doc_nodes.append(docm2m.document)

        if edge_choice == 'probability':
            weight = edge_scale_factor * docm2m.probability
        else:
            weight = edge_scale_factor * docm2m.overlap_score
        G.add_edge(docm2m.mass2motif.name,docm2m.document.name,weight = weight)
    print "Third"
    return G
def topic_pca(request,experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    context_dict = {'experiment': experiment}
    url = '/basicviz/get_topic_pca_data/' + str(experiment.id)
    context_dict['url'] = url
    return render(request,'basicviz/pca.html',context_dict)

def document_pca(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    context_dict = {}
    context_dict['experiment'] = experiment
    url = '/basicviz/get_pca_data/' + str(experiment.id)
    context_dict['url'] = url
    return render(request,'basicviz/pca.html',context_dict)

def get_topic_pca_data(request,experiment_id):


    

    experiment = Experiment.objects.get(id = experiment_id)
    motifs = Mass2Motif.objects.filter(experiment = experiment)
    features = Feature.objects.filter(experiment = experiment)

    n_motifs = len(motifs)
    n_features = len(features)


    mat = []
    motif_index = {}
    motif_pos = 0


    for motif in motifs:
        motif_index[motif] = motif_pos
        motif_pos += 1


    feature_pos = 0
    feature_index = {}

    for feature in features:
        instances = Mass2MotifInstance.objects.filter(feature = feature)
        if len(instances) > 2: # minimum to include
            feature_index[feature] = feature_pos
            new_row = [0.0 for i in range(n_motifs)]
            for instance in instances:
                motif_pos = motif_index[instance.mass2motif]
                new_row[motif_pos] = instance.probability
            feature_pos += 1
            mat.append(new_row)


    mat = np.array(mat).T

    pca = PCA(n_components = 2,whiten = True)
    pca.fit(mat)

    X = pca.transform(mat)
    pca_points = []
    for motif in motif_index:
        motif_pos = motif_index[motif]
        new_element = (X[motif_pos,0],X[motif_pos,1],motif.name,'#FF66CC')
        pca_points.append(new_element)


    max_x = np.abs(X[:,0]).max()
    max_y = np.abs(X[:,1]).max()

    factors = pca.components_
    max_factor_x = np.abs(factors[0,:]).max()
    factors[0,:] *= max_x/max_factor_x
    max_factor_y = np.abs(factors[1,:]).max()
    factors[1,:] *= max_y/max_factor_y

    pca_lines = []
    factor_colour = 'rgba(0,0,128,0.5)'
    for feature in feature_index:
        xval = factors[0,feature_index[feature]]
        yval = factors[1,feature_index[feature]]
        if abs(xval) > 0.01*max_x or abs(yval) > 0.01*max_y:
            pca_lines.append((xval,yval,feature.name,factor_colour))
    # add the weightings
    pca_data = (pca_points,pca_lines)

    return HttpResponse(json.dumps(pca_data),content_type = 'application/json')


def get_pca_data(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    theta_data = []
    documents = Document.objects.filter(experiment = experiment)
    mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
    n_mass2motifs = len(mass2motifs)
    m2mindex = {}
    msmpos = 0

    doc_m2m_threshold = get_option('doc_m2m_threshold',experiment = experiment)
    if doc_m2m_threshold:
        doc_m2m_threshold = float(doc_m2m_threshold)
    else:
        doc_m2m_threshold = 0.00 # Default value


    for document in documents:
        new_theta = [0 for i in range(n_mass2motifs)]
        dm2ms = DocumentMass2Motif.objects.filter(document = document,probability__gte = doc_m2m_threshold)
        for dm2m in dm2ms:
            if dm2m.mass2motif.name in m2mindex:
                m2mpos = m2mindex[dm2m.mass2motif.name]
            else:
                m2mpos = msmpos
                m2mindex[dm2m.mass2motif.name] = m2mpos
                msmpos += 1
            new_theta[m2mpos] = dm2m.probability
        theta_data.append(new_theta)

    pca = PCA(n_components = 2,whiten = True)
    pca.fit(theta_data)

    pca_data = []
    X = pca.transform(theta_data)
    for i in range(len(documents)):
        name = documents[i].name
        md = jsonpickle.decode(documents[i].metadata)
        color = '#ff3333'
        if 'annotation' in md:
            name = md['annotation']
            color = '#BE84CF'
        new_value = (float(X[i,0]),float(X[i,1]),name,color)
        pca_data.append(new_value)

    # pca_data = []
    return HttpResponse(json.dumps(pca_data),content_type = 'application/json')

def validation(request,experiment_id):
    experiment = Experiment.objects.get(id=experiment_id)
    context_dict = {}
    if request.method == 'POST':
        form = ValidationForm(request.POST)
        if form.is_valid():
            p_thresh = form.cleaned_data['p_thresh']
            just_annotated = form.cleaned_data['just_annotated']
            mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
            annotated_mass2motifs = []

            counts = []
            for mass2motif in mass2motifs:
                if mass2motif.annotation:
                    annotated_mass2motifs.append(mass2motif)
                    dm2ms = DocumentMass2Motif.objects.filter(mass2motif = mass2motif,probability__gte = p_thresh)
                    tot = 0
                    val = 0
                    for d in dm2ms:
                        if (just_annotated and d.document.annotation) or not just_annotated:
                            tot += 1
                            if d.validated:
                                val += 1
                    counts.append((tot,val))
            annotated_mass2motifs = zip(annotated_mass2motifs,counts)
            context_dict['annotated_mass2motifs'] = annotated_mass2motifs
            context_dict['counts'] = counts
            context_dict['p_thresh'] = p_thresh
            context_dict['just_annotated'] = just_annotated

        else:
            context_dict['validation_form'] = form
    else:

        form = ValidationForm()
        context_dict['validation_form'] = form
    context_dict['experiment'] = experiment
    return render(request,'basicviz/validation.html',context_dict)


def toggle_dm2m(request,experiment_id,dm2m_id):
    dm2m = DocumentMass2Motif.objects.get(id = dm2m_id)
    jd = []
    if dm2m.validated:
        dm2m.validated = False
        jd.append('No')
    else:
        dm2m.validated = True
        jd.append('Yes')
    dm2m.save()

    return HttpResponse(json.dumps(jd),content_type = 'application/json')
    # return validation(request,experiment_id)

def dump_validations(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    mass2motifs = Mass2Motif.objects.filter(experiment = experiment)
    annotated_mass2motifs = []
    for mass2motif in mass2motifs:
        if mass2motif.annotation:
            annotated_mass2motifs.append(mass2motif)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="valid_dump_{}.csv"'.format(experiment_id)
    writer = csv.writer(response)
    writer.writerow(['msm_id','m2m_name','m2m_annotation','doc_id','doc_annotation','valid','probability'])
    for mass2motif in annotated_mass2motifs:
        dm2ms = DocumentMass2Motif.objects.filter(mass2motif = mass2motif,probability__gte = 0.02)
        for dm2m in dm2ms:
            document = dm2m.document
            # outstring +='{},{},{},"{}",{}\n'.format(mass2motif.id,mass2motif.annotation,dm2m.document.id,dm2m.document.annotation.encode('utf8'),dm2m.validated)
            doc_name = '"' + dm2m.document.display_name + '"'
            annotation = '"' + mass2motif.annotation + '"'
            writer.writerow([mass2motif.id,mass2motif.name,mass2motif.annotation.encode('utf8'),dm2m.document.id,doc_name.encode('utf8'),dm2m.validated,dm2m.probability])

    # return HttpResponse(outstring,content_type='text')
    return response

def dump_topic_molecules(request,m2m_id):
    mass2motif = Mass2Motif.objects.get(id = m2m_id)


    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="topic_molecules_{}.csv"'.format(mass2motif.id)
    writer = csv.writer(response)
    writer.writerow(['m2m_id','m2m_name','m2m_annotation','doc_id','doc_annotation','valid','probability','doc_csid','doc_inchi'])
    
    dm2ms = DocumentMass2Motif.objects.filter(mass2motif = mass2motif,probability__gte = 0.02)
    for dm2m in dm2ms:
        document = dm2m.document
        # outstring +='{},{},{},"{}",{}\n'.format(mass2motif.id,mass2motif.annotation,dm2m.document.id,dm2m.document.annotation.encode('utf8'),dm2m.validated)
        doc_name = '"' + dm2m.document.display_name + '"'
        annotation = '"' + mass2motif.annotation + '"'
        writer.writerow([mass2motif.id,mass2motif.name,mass2motif.annotation.encode('utf8'),dm2m.document.id,doc_name.encode('utf8'),dm2m.validated,dm2m.probability,dm2m.document.csid,dm2m.document.inchikey])

    return response

def extract_docs(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    context_dict = {}
    if request.method == 'POST':
        # form has come, get the documents
        form = DocFilterForm(request.POST)
        if form.is_valid():
            all_docs = Document.objects.filter(experiment = experiment)
            selected_docs = {}
            all_m2m = Mass2Motif.objects.filter(experiment = experiment)
            annotated_m2m = []
            for m2m in all_m2m:
                if m2m.annotation:
                    annotated_m2m.append(m2m)
            for doc in all_docs:
                if form.cleaned_data['annotated_only'] and not doc.display_name:
                    # don't keep
                    continue
                else:
                    dm2m = DocumentMass2Motif.objects.filter(document = doc)
                    m2ms = [d.mass2motif for d in dm2m if d.probability > form.cleaned_data['topic_threshold']]
                    if len(list((set(m2ms) & set(annotated_m2m)))) < form.cleaned_data['min_annotated_topics']:
                        # don't keep
                        continue
                    else:
                        selected_docs[doc] = []
                        for d in dm2m:
                            if d.probability > form.cleaned_data['topic_threshold']:
                                if not form.cleaned_data['only_show_annotated']:
                                    selected_docs[doc].append(d)
                                else:
                                    if d.mass2motif.annotation:
                                        selected_docs[doc].append(d)

                        selected_docs[doc] = sorted(selected_docs[doc],key = lambda x:x.probability,reverse=True)
                context_dict['n_docs'] = len(selected_docs)
            context_dict['docs'] = selected_docs
        else:
            context_dict['doc_form'] = form
    else:
        doc_form = DocFilterForm()
        context_dict['doc_form'] = doc_form
    context_dict['experiment'] = experiment
    return render(request,'basicviz/extract_docs.html',context_dict)


def compute_overlap_score(mass2motif,document):
    # Computes the 'simon' score that looks at the proportion of the 
    # mass2motif that is represented in the document
    document_feature_instances = FeatureInstance.objects.filter(document = document)
    # Following are the phi scores
    feature_mass2motif_instances = FeatureMass2MotifInstance.objects.filter(featureinstance__in = document_feature_instances)
    score = 0.0
    for feature_mass2motif_instance in feature_mass2motif_instances:
        feature = feature_mass2motif_instance.featureinstance.feature
        m2m_feature = Mass2MotifInstance.objects.filter(mass2motif = mass2motif,feature = feature)
        if len(m2m_feature) > 0:
            score += feature_mass2motif_instance.probability * m2m_feature[0].probability
    return score

def get_option(key,experiment = None):
    # Retrieves an option, looking for an experiment specific one if it exists
    if experiment:
        options = SystemOptions.objects.filter(key = key,experiment = experiment)
        if len(options) == 0:
            options = SystemOptions.objects.filter(key = key)
    else:
        options = SystemOptions.objects.filter(key = key)
    if len(options) > 0:
        return options[0].value
    else:
        return None


def rate_by_conserved_motif_rating(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    mass2motifs = experiment.mass2motif_set.all()
    motif_scores = []

    for motif in mass2motifs:
        motif_docs = motif.documentmass2motif_set.all()
        docs = [m.document for m in motif_docs]
        total_docs = len(motif_docs)
        thresh = total_docs*0.4
        motif_features = motif.mass2motifinstance_set.all()
        n_matching = 0
        for motif_feature in motif_features:
            # How many docs is it in
            n_docs = len(FeatureInstance.objects.filter(feature = motif_feature.feature,document__in = docs))
            if n_docs > thresh:
                n_matching += 1
        motif_scores.append((motif,n_matching,len(docs)))
    
        
    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['motif_scores'] = motif_scores

    return render(request,'basicviz/rate_by_conserved_motif.html',context_dict)

def view_experiment_options(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    global_options = SystemOptions.objects.filter(experiment = None)
    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['global_options'] = global_options
    specific_options = SystemOptions.objects.filter(experiment = experiment)
    context_dict['specific_options'] = specific_options

    return render(request,'basicviz/view_experiment_options.html',context_dict)

def add_experiment_option(request,experiment_id):
    experiment = Experiment.objects.get(id = experiment_id)
    context_dict = {}
    context_dict['experiment'] = experiment
    context_dict['available'] = available_options
    if request.method == 'POST':
        option_form = SystemOptionsForm(request.POST)
        if option_form.is_valid():
            new_option = option_form.save(commit = False)
            new_option.experiment = experiment
            new_option.save()

            

            return view_experiment_options(request,experiment.id)

        else:
            context_dict['option_form'] = option_form
    else:
        option_form = SystemOptionsForm()
        context_dict['option_form'] = option_form

    
    return render(request,'basicviz/add_experiment_option.html',context_dict)

def delete_experiment_option(request,option_id):
    option = SystemOptions.objects.get(id = option_id)
    experiment = option.experiment
    option.delete()
    return view_experiment_options(request,experiment.id)

def edit_experiment_option(request,option_id):
    option = SystemOptions.objects.get(id = option_id)
    experiment = option.experiment

    if request.method == 'POST':
        option_form = SystemOptionsForm(request.POST,instance = option)
        if option_form.is_valid():
            option_form.save()
            return view_experiment_options(request,experiment.id)
        else:
            context_dict['option_form'] = option_form
    else:
        option_form = SystemOptionsForm(instance = option)
        context_dict = {}
        context_dict['experiment'] = experiment
        context_dict['option_form'] = option_form
        context_dict['option'] = option
        context_dict['available'] = available_options
    return render(request,'basicviz/edit_experiment_option.html',context_dict)   

def view_mf_experiment_options(request,mfe_id):
    mfe = MultiFileExperiment.objects.get(id = mfe_id)
    links = mfe.multilink_set.all()
    individuals = [l.experiment for l in links]
    first_experiment = individuals[0]

    global_options = SystemOptions.objects.filter(experiment = None)
    specific_options = SystemOptions.objects.filter(experiment = first_experiment)

    context_dict = {}
    context_dict['mfe'] = mfe
    context_dict['global_options'] = global_options
    context_dict['specific_options'] = specific_options

    return render(request,'basicviz/view_mf_experiment_options.html',context_dict)

def add_mf_experiment_option(request,mfe_id):
    mfe = MultiFileExperiment.objects.get(id = mfe_id)
    context_dict = {}
    context_dict['mfe'] = mfe
    context_dict['available'] = available_options
    links = mfe.multilink_set.all()
    individuals = [l.experiment for l in links]

    if request.method == 'POST':
        option_form = SystemOptionsForm(request.POST)
        if option_form.is_valid():
            for experiment in individuals:
                key = option_form.cleaned_data['key']
                value = option_form.cleaned_data['value']
                new_option = SystemOptions.objects.get_or_create(experiment = experiment,key = key)[0]
                new_option.value = value
                new_option.save()

            

            return view_mf_experiment_options(request,mfe.id)

        else:
            context_dict['option_form'] = option_form
    else:
        option_form = SystemOptionsForm()
        context_dict['option_form'] = option_form

    
    return render(request,'basicviz/add_mf_experiment_option.html',context_dict)

def delete_mf_experiment_option(request,option_id):
    option = SystemOptions.objects.get(id = option_id)
    experiment = option.experiment
    link = experiment.multilink_set.all()
    mfe = link[0].multifileexperiment
    links = mfe.multilink_set.all()
    individuals = [l.experiment for l in links]
    key = option.key
    for experiment in individuals:
        option = SystemOptions.objects.get(experiment = experiment,key = key)
        option.delete()

    return view_mf_experiment_options(request,mfe.id)

def edit_mf_experiment_option(request,option_id):
    option = SystemOptions.objects.get(id = option_id)
    experiment = option.experiment
    link = experiment.multilink_set.all()
    mfe = link[0].multifileexperiment
    context_dict = {}
    context_dict['mfe'] = mfe
    context_dict['option'] = option
    context_dict['available'] = available_options
    links = mfe.multilink_set.all()
    individuals = [l.experiment for l in links]
    if request.method == 'POST':
        option_form = SystemOptionsForm(request.POST)
        if option_form.is_valid():
            key = option_form.cleaned_data['key']
            value = option_form.cleaned_data['value']
            for experiment in individuals:
                new_option = SystemOptions.objects.get_or_create(experiment = experiment,key = key)[0]
                new_option.value = value
                new_option.save()
            return view_mf_experiment_options(request,mfe.id)
        else:
            context_dict['option_form'] = option_form
    else:
        option_form = SystemOptionsForm(instance = option)
        context_dict['option_form'] = option_form

    return render(request,'basicviz/edit_mf_experiment_option.html',context_dict)

def alpha_de(request,mfe_id):
    mfe = MultiFileExperiment.objects.get(id = mfe_id)
    context_dict = {'mfe':mfe}
    links = mfe.multilink_set.all()
    individuals = [l.experiment for l in links]
    tu = zip(individuals,individuals)
    tu = sorted(tu,key = lambda x: x[0].name)
    if request.method == 'POST':
        form = AlphaDEForm(tu,request.POST)
        if form.is_valid():
            group1_experiments = form.cleaned_data['group1']
            group2_experiments = form.cleaned_data['group2']
            motifs = individuals[0].mass2motif_set.all().order_by('name')

            if mfe.alpha_matrix:
                alp_vals_with_names = jsonpickle.decode(mfe.alpha_matrix)
                alp_vals = []
                for av in alp_vals_with_names:
                    newav = av[2:-1]
                    alp_vals.append(newav)

            else:
                alp_vals = make_alpha_matrix(individuals,normalise = True)

            group1_index = []
            group2_index = []
            motif_scores = []
            for experiment_name in group1_experiments:
                experiment = Experiment.objects.get(name = experiment_name)
                group1_index.append(individuals.index(experiment))
            for experiment_name in group2_experiments:
                experiment = Experiment.objects.get(name = experiment_name)
                group2_index.append(individuals.index(experiment))
            for i,alp in enumerate(alp_vals):
                a = np.array(alp)
                g1 = a[group1_index]
                g2 = a[group2_index]
                de = (g1.mean()-g2.mean())/(g1.std() + g2.std())
                t, p = ttest_ind(g1,g2,equal_var = False)
                motif_scores.append((motifs[i],de,p))
            context_dict['motif_scores'] = motif_scores

                
                        
        else:
            context_dict['alpha_de_form'] = form
    else:
        form = AlphaDEForm(tu)        
        context_dict['alpha_de_form'] = form
    return render(request,'basicviz/alpha_de.html',context_dict)
