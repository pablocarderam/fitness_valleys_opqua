
'''
Cardenas & Santos-Vega, 2021
Coded by github.com/pablocarderam
Creates heatmaps of contact rate and mutant fitness cost used in Figure 3e-i
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

# each mutation away from the WT sequence implies a drop in fitness
def fitnessDropFunction(genome,drop):
    return np.power( drop, td.hamming(genome,wt_genome) )

model = Model()

model.newSetup(
    'base_setup', preset='host-host',
    possible_alleles='ARNDCEQGHILKMFPSTWYV',
    num_loci=len(wt_genome),
    mean_inoculum_host=1,
    mutate_in_host=5e-2,
    contact_rate_host_host=1.15e-1,
    recombine_in_host=0
    )

num_hosts = 500
tf = 1000

samples = 20 # width of heatmap in pixels
replicates = 10

fitness_drops = np.around(
    np.linspace( 1, 0, samples, endpoint=True ), decimals=3
    )
# make a list of fitness functions to feed into runParamSweep
fitness_functions = [
    lambda genome,d=val: fitnessDropFunction(genome,d)
        for val in fitness_drops
    ]
contact_rates = np.around(
    np.linspace( 2e-1, 1e-1, samples, endpoint=True ), decimals=3
    )

model.newPopulation(
    'my_population','base_setup', num_hosts=num_hosts, num_vectors=num_hosts
    )
model.addPathogensToHosts( 'my_population', {wt_genome:int(num_hosts*0.5)} )

# To generate data, uncomment the following...
# Run simulations
df,res = model.runParamSweep(
    0,tf,'base_setup',
    param_sweep_dic = {
        'contact_rate_host_host' : contact_rates,
        'fitnessHost' : fitness_functions
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
pwd_genomes = []
    # mean pairwise distances between genomes among pathogens at end

inf_hosts = []

for mod in res:
    dat = mod.saveToDataFrame('tmp.csv', verbose=0)
        # full data is overwritten every iteration to save on hard disk space,
        # my computer is kinda running on fumes here. Probably makes sense to
        # save everything.
    comp = compositionDf(
        dat, track_specific_sequences=[wt_genome], num_top_sequences=-1
        )
    total_infected_hosts = compartmentDf(dat)['Infected'].sum()
    total_infections = comp.drop(columns=['Time']).to_numpy().sum()
    num_wt = comp[wt_genome].sum() if wt_genome in comp.columns else 0
    non_wt = total_infections - num_wt if total_infected_hosts > 1 else 0
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

    end_genomes = list(set([
        g for h in mod.populations['my_population'].hosts
            for g in h.pathogens.keys()
        ]))
    mean_pwd = np.mean([
        td.hamming( g1,g2 ) for g2 in end_genomes for g1 in end_genomes
        ])
    pwd_genomes.append(mean_pwd)

    inf_hosts.append(total_infected_hosts/num_hosts)

dat = df
dat['mutant_frac'] = mut_frac
dat['mut_frac_inf_hosts'] = mut_frac_inf_hosts
dat['num_genomes'] = num_genomes
dat['max_gen_distance'] = max_gen_distance
dat['gen_variance'] = gen_variance
dat['ssq_genomes'] = ssq_genomes
dat['infected_hosts'] = inf_hosts
dat['pwd_genomes'] = pwd_genomes
dat['fitnessHost'] = np.tile(
    fitness_drops,
    int( len(res) / len(fitness_drops) )
    )
dat['mut_frac_hosts'] = dat['mut_frac_inf_hosts'] * dat['infected_hosts']
dat.to_csv('contact_fitness_heatmap.csv')
# ...until here

# Reformat data for heatmaps
dat = pd.concat([pd.read_csv('contact_fitness_heatmap.csv')]).fillna(0)

dat = dat.groupby(['contact_rate_host_host','fitnessHost']).mean().reset_index()
dat['fitness_cost'] = 1 - dat['fitnessHost']
dat['fitness_cost'] = dat['fitness_cost'].round(decimals=3)
dat['contact_rate_host_host'] = dat['contact_rate_host_host'].round(decimals=3)
dat_frac = dat.pivot('contact_rate_host_host','fitness_cost','mutant_frac')
dat_mfih = dat.pivot('contact_rate_host_host','fitness_cost','mut_frac_hosts')
dat_ngen = dat.pivot('contact_rate_host_host','fitness_cost','num_genomes')
dat_mgen = dat.pivot('contact_rate_host_host','fitness_cost','max_gen_distance')
dat_mpwd = dat.pivot('contact_rate_host_host','fitness_cost','pwd_genomes')
dat_vgen = dat.pivot('contact_rate_host_host','fitness_cost','gen_variance')
dat_sgen = dat.pivot('contact_rate_host_host','fitness_cost','ssq_genomes')
dat_ihos = dat.pivot('contact_rate_host_host','fitness_cost','infected_hosts')

# Plot heatmaps
plt.rcParams.update({'font.size': 12})

def plotHeatmap(
        dat,cmap_lab,file_name,cmap, vmin=None, vmax=None, show_labels=False):
    plt.figure(figsize=(8,8), dpi=200)
    ax = plt.subplot(1, 1, 1)
    ax = sns.heatmap(
        dat, linewidth = 0 , annot = False, cmap=cmap,
        cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
        xticklabels=show_labels, yticklabels=show_labels
        )
    ax.figure.axes[-1].yaxis.label.set_size(15)
    ax.invert_yaxis()
    spacing = '\n\n\n' if show_labels else ''
    plt.xlabel(
        'Competitive disadvantage \nof each additional mutation'+spacing,
        fontsize=15
        )
    plt.ylabel('Host-host contact rate'+spacing,fontsize=15)
    plt.savefig(file_name, bbox_inches='tight')

plotHeatmap(
    dat_frac,'Mutant prevalence in pathogen population',
    'pathogen_heatmap_stochastic.png','rocket', vmin=0, vmax=1
    )
plotHeatmap(
    dat_mfih,'Fraction of hosts \nwith only mutant pathogens',
    'host_heatmap_stochastic.png','mako', vmin=0, vmax=None
    )
plotHeatmap(
    dat_ngen,'Number of unique genomes explored',
    'num_genomes_heatmap.png','viridis', vmin=0, vmax=None
    )
plotHeatmap(
    dat_mgen,'Maximum distance from wild-type',
    'max_gen_distance_heatmap.png','inferno', vmin=0, vmax=None
    )
plotHeatmap(
    dat_mpwd,'Mean genome pairwise distance at end',
    'pwise_distance_heatmap.png','plasma', vmin=0, vmax=None
    )
