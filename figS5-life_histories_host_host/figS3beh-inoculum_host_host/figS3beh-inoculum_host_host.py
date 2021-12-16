
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

# each mutation away from the WT sequence implies a drop in fitness
def fitnessDropFunction(genome,drop):
    return np.power( drop, td.hamming(genome,wt_genome) )

model = Model()

num_hosts = 500
tf = 1000 # final time
samples = 10 # width of heatmap in pixels
replicates = 32

inoculums = np.linspace( 10, 1, samples )
mut_rates = np.around( np.linspace( 5e-2, 0, samples, endpoint=False ), decimals=3 )
# By starting with the larger values, we do the longer simulations first

model.newSetup(
    'base_setup', preset='host-host',
    possible_alleles='ARNDCEQGHILKMFPSTWYV',
    num_loci=len(wt_genome),
    fitnessHost=lambda g:fitnessDropFunction(g,0.5), # fix drop to 50% drop
    mean_inoculum_host=1,
    mutate_in_host=0.05,
    contact_rate_host_vector=1.25e-1,
    recombine_in_host=0,
    recombine_in_vector=0
    )

model.newPopulation(
    'my_population','base_setup', num_hosts=num_hosts
    )
model.addPathogensToHosts( 'my_population', {wt_genome:50} )

# To generate data, uncomment the following...
# Run simulations
df,res = model.runParamSweep(
    0,tf,'base_setup',
    param_sweep_dic = {
        'mean_inoculum_host': inoculums,
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
dat.to_csv('inoculum_sweep.csv')
# ...until here

dat = pd.concat(
    [pd.read_csv('inoculum_sweep.csv')]
    ).fillna(0)

dat['mutant_frac'] = dat['mutant_frac']*100
dat['num_genomes'] = dat['num_genomes']*1e-3

# Plot heatmaps
plt.rcParams.update({'font.size': 20})

sns.set_style("ticks")
plt.figure(figsize=(6,6), dpi=300)
ax = sns.lineplot(
    x = 'mean_inoculum_host', y='mutant_frac', data=dat,
    color="#ff5656", marker='o',ci=95
    )
ax.set_xlabel("Mean inoculum (pathogens)")
ax.set_ylabel('Mutants (% of all pathogens)')
ax.set_ylim(bottom=0, top=100)
# ax.set_xlim(left=-50, right=1050)
plt.tight_layout()
plt.savefig('inoculum_sweep_patfrac.png', bbox_inches='tight')

plt.figure(figsize=(6,6), dpi=300)
ax = sns.lineplot(
    x = 'mean_inoculum_host', y='num_genomes',
    data=dat, color="#2271B2", marker='o',ci=95
    )
ax.set_xlabel("Mean inoculum (pathogens)")
ax.set_ylabel('Unique genomes (thousands)')
ax.set_ylim(bottom=0, top=21)
# ax.set_xlim(left=-50, right=1050)
plt.tight_layout()
plt.savefig('inoculum_sweep_numgen.png', bbox_inches='tight')

plt.figure(figsize=(6,6), dpi=300)
ax = sns.lineplot(
    x = 'mean_inoculum_host', y='max_gen_distance',
    data=dat, color="#e69f00", marker='o',ci=95
    )
ax.set_xlabel("Mean inoculum (pathogens)")
ax.set_ylabel('Maximum genome distance from WT')
ax.set_ylim(bottom=2.8, top=6.2)
# ax.set_xlim(left=-50, right=1050)
plt.tight_layout()
plt.savefig('inoculum_sweep_gendis.png', bbox_inches='tight')
