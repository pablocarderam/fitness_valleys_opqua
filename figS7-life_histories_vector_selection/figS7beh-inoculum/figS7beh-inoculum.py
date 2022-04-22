
'''
Cardenas & Santos-Vega, 2021
Coded by github.com/pablocarderam
Creates heatmaps of parameter sweeps for mutation in and inoculation from
vectors
used in Figure 5b,e,h
'''

import copy as cp
import textdistance as td

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plots
import seaborn as sns # pretty plots

from opqua.model import Model
from opqua.internal.data import compositionDf, compartmentDf

wt_genome = '\
MEGEKVKTKANSISNFSMTYDRESGGNSNSDDKSGSSSENDSNSFMNLTSDKNEKTENNSFLLNNSSYGNVKDSLLESID\
MSVLDSNFDSKKDFLPSNLSRTFNNMSKDNIGNKYLNKLLNKKKDTITNENNNINHNNNNNNLTANNITNNLINNNMNSP\
SIMNTNKKENFLDAANLINDDSGLNNLKKFSTVNNVNDTYEKKIIETELSDASDFENMVGDLRITFINWLKKTQMNFIRE\
KDKLFKDKKELEMERVRLYKELENRKNIEEQKLHDERKKLDIDISNGYKQIKKEKEEHRKRFDEERLRFLQEIDKIKLVL\
YLEKEKYYQEYKNFENDKKKIVDANIATETMIDINVGGAIFETSRHTLTQQKDSFIEKLLSGRHHVTRDKQGRIFLDRDS\
ELFRIILNFLRNPLTIPIPKDLSESEALLKEAEFYGIKFLPFPLVFCIGGFDGVEYLNSMELLDISQQCWRMCTPMSTKK\
AYFGSAVLNNFLYVFGGNNY'

# Generate two fitness peaks a set mutational distance away:
peak_distance_from_wt = 3
amino_acids = 'ARNDCEQGHILKMFPSTWYV'
peak_genome_host = wt_genome
peak_genome_vector = wt_genome

for i in range(peak_distance_from_wt):
    mut = wt_genome[i]
    while mut == wt_genome[i] or mut == wt_genome[-(i+1)]:
        mut = np.random.choice(list(amino_acids))

    peak_genome_host = peak_genome_host[0:i] \
        + mut + peak_genome_host[i+1:]
    peak_genome_vector = peak_genome_vector[0:len(peak_genome_vector)-i-1] \
        + mut + peak_genome_vector[len(peak_genome_vector)-i:]

# each mutation away from the peak sequence implies a drop in fitness
def fitnessDropFunction(genome,drop,peak_genome):
    return np.power( drop, td.hamming(genome,peak_genome) )

model = Model()

num_hosts = 250
tf = 1000 # final time
samples = 20 # width of heatmap in pixels
replicates = 10

inoculums = np.linspace( 10, 1, samples )
mut_rates = np.linspace( 2e-2, 0, samples, endpoint=True )
# By starting with the larger values, we do the longer simulations first

model.newSetup(
    'base_setup', preset='vector-borne',
    possible_alleles='ARNDCEQGHILKMFPSTWYV',
    num_loci=len(wt_genome),
    fitnessHost=lambda g:fitnessDropFunction(g,0.5,peak_genome_host),
    fitnessVector=lambda g:fitnessDropFunction(g,0.5,peak_genome_vector),
    mean_inoculum_host=1,
    mean_inoculum_vector=1,
    mutate_in_host=0.01,
    mutate_in_vector=0.01,
    contact_rate_host_vector=1.25e-1,
    recombine_in_host=0,
    recombine_in_vector=0
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
        'mean_inoculum_host': inoculums,
        'mean_inoculum_vector': inoculums,
        # 'mutate_in_vector': mut_rates,
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
dat.to_csv('inoculum_inoculum_heatmap.csv')
# ...until here

dat = pd.concat(
    [pd.read_csv('inoculum_inoculum_heatmap.csv')]
    ).fillna(0)

dat['mean_inoculum_vector'] = dat['mean_inoculum_vector'].round(decimals=4)
dat['mean_inoculum_host'] = dat['mean_inoculum_host'].round(decimals=4)

dat['num_genomes'] = dat['num_genomes'] / 1000
# dat['mutant_frac'] = dat['mutant_frac'] * 100

# Reformat data for heatmaps
dat = dat.groupby(
    ['mean_inoculum_host','mean_inoculum_vector']
    ).mean().reset_index()
dat_frac = dat.pivot('mean_inoculum_host','mean_inoculum_vector','mutant_frac')
dat_ngen = dat.pivot('mean_inoculum_host','mean_inoculum_vector','num_genomes')
dat_mgen = dat.pivot('mean_inoculum_host','mean_inoculum_vector','max_gen_distance')
dat_vgen = dat.pivot('mean_inoculum_host','mean_inoculum_vector','gen_variance')
dat_sgen = dat.pivot('mean_inoculum_host','mean_inoculum_vector','ssq_genomes')
dat_ihos = dat.pivot('mean_inoculum_host','mean_inoculum_vector','infected_hosts')
dat_mfih = dat.pivot('mean_inoculum_host','mean_inoculum_vector','mut_frac_hosts')

# Plot heatmaps
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
    # plt.xlabel('Mean inoculum into host (pathogens)',fontsize=15)
    # plt.ylabel('Mean inoculum into vector (pathogens)',fontsize=15)
    plt.savefig(file_name, bbox_inches='tight')

plotHeatmap(
    dat_frac,'Mutant prevalence in pathogen population',
    'inoculum_pathogen_heatmap.png','rocket', vmin=0.2, vmax=0.85
    )
plotHeatmap(
    dat_mfih,'Fraction of hosts \nwith only mutant pathogens',
    'inoculum_host_heatmap_stochastic.png','mako', #vmin=0, vmax=None
    )
plotHeatmap(
    dat_ngen,'Number of genomes explored',
    'inoculum_num_genomes_heatmap.png','viridis', vmin=0.4, vmax=1.5
    )
plotHeatmap(
    dat_mgen,'Maximum distance from wild-type',
    'inoculum_max_gen_distance_heatmap.png','inferno', vmin=3, vmax=5.5
    )
plotHeatmap(
    dat_ihos,'Infected host fraction',
    'inoculum_inf_hosts_heatmap.png','cividis'
    )
plotHeatmap(
    dat_vgen,'Variance of distance from wild-type',
    'inoculum_gen_variance_heatmap.png','magma'
    )
plotHeatmap(
    dat_sgen,'Sum of squares of distance from wild-type',
    'inoculum_ssq_gen_distance_heatmap.png','magma'
    )
