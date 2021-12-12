
'''
Cardenas & Santos-Vega, 2021
Coded by github.com/pablocarderam
Creates plot of contact rate sweep for survival used in Figure 2c
'''

import textdistance as td

import numpy as np
import pandas as pd
import joblib as jl
import matplotlib.pyplot as plt # plots
import seaborn as sns # pretty plots

from opqua.model import Model
from opqua.internal.data import compositionDf, compartmentDf

CB_PALETTE = ["#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
    # www.cookbook-r.com/Graphs/Colors_(ggplot2)/#a-colorblind-friendly-palette
    # http://jfly.iam.u-tokyo.ac.jp/color/

wt_seq = 'AAAAAAAA' # wild-type genome sequence
re_seq = 'BBBBBBBB' # drug resistant mutant genome sequence

# Defines intra-host competitive fitness of pathogens
def fitnessLandscape(genome):
    num_b = genome.count('B')
    if num_b < 3:
        f = 1 / ( 2 * ( (num_b+1)**5 ) )
    else:
        f = 1 / ( ( len(genome)-num_b+1 )**3 )

    return f

# Modifies recovery rate of resistant pathogens, e.g. if hosts are seeking
# treatment by a drug when they get sick and the resistant pathogen endures for
# longer.
# Included in example as a way of increasing mutant persistance, makes effect
# easier to visualize.
def drugCheck(genome,re_seq='BBBBBBBB'):
    return 1 if genome!=re_seq else 0.75

# Parameters
num_hosts = 1000
contact_rates = np.concatenate( # defines sweep
    [ np.linspace( 4e-3,1e-2,15 ), np.linspace( 1e-2,2e-2,10 ) ]
    )

num_replicates = 100 # per contact rate
t_d = 10000 # time at which drug is applied

model = Model()
model.newSetup( # Now, we'll define our new setup:
    'base_setup', preset='host-host', # Use default host-host parameters.
    possible_alleles='AB',
    num_loci=len(wt_seq),
    fitnessHost=fitnessLandscape,
    recoveryHost=drugCheck,
    mean_inoculum_host=1,
    contact_rate_host_host=1e-2,
    recovery_rate_host=5e-3,
    mutate_in_host=2e-2,
    recombine_in_host=0
    )

model.newPopulation('my_population','base_setup', num_hosts=num_hosts)
model.addPathogensToHosts(
    'my_population',
    { wt_seq:int(num_hosts*0.8) }
        # start all simulations from a high number of infections
    )

model.newIntervention( t_d, 'treatHosts', [ 'my_population', 1, [re_seq] ] )

t_f = t_d+500 # final time

df,res = model.runParamSweep(
    0,t_f,'base_setup',
    param_sweep_dic = {
        'contact_rate_host_host' : contact_rates,
        },
    replicates=num_replicates,host_sampling=0
    )

# Gather data into lists and make a dataframe
mut_frac = [] # mutant fraction of pathogen population
inf_hosts = [] # number of hosts infected

for mod in res:
    dat = mod.saveToDataFrame('tmp.csv', verbose=0)
        # full data is overwritten every iteration to save on hard disk space,
        # my computer is kinda running on fumes here. Probably makes sense to
        # save everything.
    comp = compositionDf(
        dat, track_specific_sequences=[wt_seq], num_top_sequences=2
        )
    total_infected_hosts = compartmentDf(dat)['Infected'].sum()
    total_infections = comp.drop(columns=['Time']).to_numpy().sum()
    num_wt = comp[wt_seq].sum() if wt_seq in comp.columns else 0
    non_wt = total_infections - num_wt
    mut_frac.append( non_wt / max(total_infections,1) ) # avoid division by zero
    inf_hosts.append(total_infected_hosts/num_hosts)

dat = df
dat['mutant_frac'] = mut_frac
dat['infected_hosts'] = inf_hosts
dat['Alive'] = dat['infected_hosts'] > 0

dat.to_csv('contact_sweep_survival.csv')

# Plot
sns.set_style("ticks")
plt.figure(figsize=(8,4), dpi=300)
ax = sns.lineplot(
    x = 'contact_rate_host_host', y='Alive', data=dat,
    color="brown", marker='o',ci=95
    )
ax.set_xlabel("Contact rate (1/unit time)")
ax.set_ylabel(
    'Probability of resistant \npathogens surviving treatment)')
plt.tight_layout()
plt.savefig('Fig2_replicate_survival.png', bbox_inches='tight')


# Plot two separate runs of 100 replicates:
dat = pd.concat( [pd.read_csv('contact_sweep_survival.csv'), pd.read_csv('contact_sweep_survival_2.csv') ] )

sns.set_style("ticks")
plt.figure(figsize=(8,3.5), dpi=300)
ax = sns.lineplot(x = 'contact_rate_host_host', y='Alive', data=dat, color="#003a2b", marker='o',ci=95)#, join=False, markers="_")
ax.set_xlabel("Contact rate (1/unit time)")
ax.set_ylabel('Probability of pathogens surviving treatment')
plt.tight_layout()
plt.savefig('Fig2_replicate_survival.png', bbox_inches='tight')
