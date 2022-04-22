
'''
Cardenas & Santos-Vega, 2021
Coded by github.com/pablocarderam
Creates heatmaps of parameter sweeps for recombination in hosts and vectors
used in Figure 5c,f,i
'''

import copy as cp
import textdistance as td

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plots
import seaborn as sns # pretty plots

from opqua.model import Model
from opqua.internal.data import compositionDf, compartmentDf, pathogenDistanceDf

wt_genome = '\
MEGEKVKTKANSISNFSMTYDRESGGNSNSDDKSGSSSENDSNSFMNLTSDKNEKTENNSFLLNNSSYGNVKDSLLESID\
MSVLDSNFDSKKDFLPSNLSRTFNNMSKDNIGNKYLNKLLNKKKDTITNENNNINHNNNNNNLTANNITNNLINNNMNSP\
SIMNTNKKENFLDAANLINDDSGLNNLKKFSTVNNVNDTYEKKIIETELSDASDFENMVGDLRITFINWLKKTQMNFIRE\
KDKLFKDKKELEMERVRLYKELENRKNIEEQKLHDERKKLDIDISNGYKQIKKEKEEHRKRFDEERLRFLQEIDKIKLVL\
YLEKEKYYQEYKNFENDKKKIVDANIATETMIDINVGGAIFETSRHTLTQQKDSFIEKLLSGRHHVTRDKQGRIFLDRDS\
ELFRIILNFLRNPLTIPIPKDLSESEALLKEAEFYGIKFLPFPLVFCIGGFDGVEYLNSMELLDISQQCWRMCTPMSTKK\
AYFGSAVLNNFLYVFGGNNY'

# each mutation away from the WT sequence implies a drop in fitness
def fitnessDropFunction(genome,drop):
    return np.power( drop, td.hamming(genome,wt_genome) )

model = Model()

num_hosts = 250
tf = 1000 # final time
samples = 20 # width of heatmap in pixels
replicates = 10

recomb_rates = np.linspace( 1e0, 0, samples )
# By starting with the larger values, we do the longer simulations first

model.newSetup(
    'base_setup', preset='vector-borne',
    possible_alleles='ARNDCEQGHILKMFPSTWYV',
    num_loci=len(wt_genome),
    fitnessHost=lambda g:fitnessDropFunction(g,0.5),
    contact_rate_host_vector=1.25e-1,
    mean_inoculum_host=10,
    mean_inoculum_vector=10,
    mutate_in_host=0.01,
    mutate_in_vector=0.01,
    recombine_in_host=0,
    recombine_in_vector=0,
    num_crossover_host=5,
    num_crossover_vector=5
        # average of 5 crossover events in every recombination event
    )

model.newPopulation(
    'my_population','base_setup', num_hosts=num_hosts, num_vectors=num_hosts
    )
model.addPathogensToHosts( 'my_population', {wt_genome:50} )

# To generate data, uncomment the following...
# Run simulations
df,res = model.runParamSweep(
    0,tf,'base_setup',
    param_sweep_dic = {
        # 'contact_rate_host_vector' : contact_rates,
        # 'fitnessHost' : [fitness_functions[0]],#fitness_functions
        # 'mean_inoculum_vector': inoculums,
        # 'mutate_in_host': mut_rates,
        'recombine_in_vector':recomb_rates,
        'recombine_in_host':recomb_rates,
        # 'num_crossover_vector':num_cross,
        },
    replicates=replicates,host_sampling=0,vector_sampling=1000
    )

# Gather data into lists and make a dataframe
mut_frac = [] # mutant fraction of pathogen population
mut_frac_inf_hosts = [] # mutant-only fraction of infected host population
num_genomes = [] # number of unique genomes explored
max_gen_distance = [] # max of distances from WT among pathogens at end
gen_variance = [] # variance of distances from WT among pathogens at end
ssq_genomes = [] # sum of squares of distances from WT among pathogens at end
inf_hosts = [] # number of infected hosts

for mod in res:
    dat = mod.saveToDataFrame('tmp.csv', verbose=0)
        # full data is overwritten every iteration to save on hard disk space,
        # my computer is kinda running on fumes here. Probably makes sense to
        # save everything.
    comp = compositionDf(
        dat, track_specific_sequences=[wt_genome], num_top_sequences=2
        )
    total_infected_hosts = compartmentDf(dat)['Infected'].sum()
    total_infections = max( comp.drop(columns=['Time']).to_numpy().sum(), 0 )
    num_wt = comp[wt_genome].sum() if wt_genome in comp.columns else 0
    non_wt = total_infections - num_wt
    non_wt_hosts = total_infected_hosts - num_wt if total_infected_hosts > 1 \
        else 0
    mut_frac.append( non_wt / total_infections )
    mut_frac_inf_hosts.append( non_wt_hosts / total_infected_hosts )

    distances = np.array([
        td.hamming( g,wt_genome ) for g in mod.global_trackers['genomes_seen']
        ])
    max_d = distances.max()
    var_d = np.var(distances)
    ssq_d = np.sum(distances**2)
    num_genomes.append( len(distances) )
    max_gen_distance.append( max_d )
    gen_variance.append( var_d )
    ssq_genomes.append( ssq_d )
    inf_hosts.append(
        total_infected_hosts / len( mod.populations['my_population'].hosts )
        )

dat = df
dat['mutant_frac'] = mut_frac
dat['num_genomes'] = num_genomes
dat['max_gen_distance'] = max_gen_distance
dat['gen_variance'] = gen_variance
dat['ssq_genomes'] = ssq_genomes
dat['infected_hosts'] = inf_hosts
dat['mut_frac_inf_hosts'] = mut_frac_inf_hosts
dat['mut_frac_hosts'] = dat['mut_frac_inf_hosts'] * dat['infected_hosts']
dat.to_csv('dat_heatmap_recombination.csv')
# ...until here

dat = pd.concat([
    pd.read_csv('dat_heatmap_recombination.csv')
    ]).fillna(0)

dat['num_genomes'] = dat['num_genomes'] / 1000
# dat['mutant_frac'] = dat['mutant_frac'] * 100

# Reformat data for heatmaps
dat = dat.groupby(
    ['recombine_in_vector','recombine_in_host']
    ).mean().reset_index()
dat['recombine_in_vector'] = dat['recombine_in_vector'].round(decimals=3)
dat['recombine_in_host'] = dat['recombine_in_host'].round(decimals=3)
dat_frac = dat.pivot('recombine_in_host','recombine_in_vector','mutant_frac')
dat_ngen = dat.pivot('recombine_in_host','recombine_in_vector','num_genomes')
dat_mgen = dat.pivot(
    'recombine_in_host','recombine_in_vector','max_gen_distance'
    )
dat_vgen = dat.pivot('recombine_in_host','recombine_in_vector','gen_variance')
dat_sgen = dat.pivot('recombine_in_host','recombine_in_vector','ssq_genomes')
dat_ihos = dat.pivot('recombine_in_host','recombine_in_vector','infected_hosts')

plt.rcParams.update({'font.size': 12})

def plotHeatmap(dat,cmap_lab,file_name,cmap, vmin=None, vmax=None):
    plt.figure(figsize=(8,6.45), dpi=200)
    ax = plt.subplot(1, 1, 1)
    ax = sns.heatmap(
        dat, linewidth = 0 , annot = False, cmap=cmap,
        cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax
        )
    ax.figure.axes[-1].yaxis.label.set_size(15)
    ax.invert_yaxis()
    # plt.xlabel('Recombination rate in vector (1/t)',fontsize=15)
    # plt.ylabel('Recombination rate in host (1/t)',fontsize=15)
    plt.savefig(file_name, bbox_inches='tight')

plotHeatmap(
    dat_frac,'Mutant fraction of pathogen population',
    'rec_pathogen_heatmap.png','rocket', vmin=0.5, vmax=1
    )
plotHeatmap(
    dat_ngen,'Number of unique genomes explored',
    'rec_genomes_heatmap.png','viridis', vmin=0.5, vmax=2
    )
plotHeatmap(
    dat_mgen,'Maximum genome distance from wild-type',
    'rec_max_gen_distance_heatmap.png','inferno', vmin=4, vmax=8
    )
plotHeatmap(
    dat_ihos,'Infected host fraction',
    'rec_inf_hosts_heatmap.png','cividis'
    )
plotHeatmap(
    dat_vgen,'Variance of distance from wild-type',
    'rec_gen_variance_heatmap.png','magma'
    )
plotHeatmap(
    dat_sgen,'Sum of squares of distance from wild-type',
    'rec_ssq_gen_distance_heatmap.png','magma'
    )
