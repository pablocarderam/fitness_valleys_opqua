
'''
Cardenas & Santos-Vega, 2021
Coded by github.com/pablocarderam
Creates plot of contact rate sweep for fraction of hosts with only mutant
pathogens used in Figure 2e
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

# Parameters
num_hosts = 1000
contact_rates = np.concatenate( # defines sweep
    [ np.linspace( 4e-3,1e-2,15 ), np.linspace( 1e-2,2e-2,10 ) ]
    )
recovery_rate = 5e-3 # used later for plotting too

num_replicates = 100 # per contact rate
t_f = 10000 # final time point

# function to kill pathogens with resistant genome in this example
def lethalCheck(genome,lethal_genomes=[ re_seq ]):
    return 0 if genome in lethal_genomes else 1

model = Model()
model.newSetup( # Now, we'll define our new setup:
    'base_setup', preset='host-host', # Use default host-host parameters.
    possible_alleles='AB',
    num_loci=len(wt_seq),
    fitnessHost=fitnessLandscape,
    contactHost=lethalCheck,
    mutationHost=lethalCheck,
    mean_inoculum_host=1,
    contact_rate_host_host=1e-2,
    recovery_rate_host=recovery_rate,
    mutate_in_host=2e-2,
    recombine_in_host=0
    )

model.newPopulation('my_population','base_setup', num_hosts=num_hosts)
model.addPathogensToHosts(
    'my_population',
    { wt_seq:int(num_hosts*0.8) }
        # start all simulations from a high number of infections
    )

df,res = model.runParamSweep(
    0,t_f,'base_setup',
    param_sweep_dic = {
        'contact_rate_host_host' : contact_rates,
        },
    replicates=num_replicates,host_sampling=0
    )

# Gather data into lists and make a dataframe
mut_frac = [] # mutant-only fraction of infected host population
    # (different from before!)
inf_hosts = [] # number of hosts infected

for mod in res:
    dat = mod.saveToDataFrame('tmp.csv', verbose=0)
        # full data is overwritten every iteration to save on hard disk space,
        # my computer is kinda running on fumes here. Probably makes sense to
        # save everything.
    comp = compositionDf(
        dat, track_specific_sequences=[wt_seq], num_top_sequences=6,
        count_individuals_based_on_model=model
            # this argument means only one pathogen per host will be counted
            # (the one with greatest fitness, so WT if present)
        )
    total_infected_hosts = compartmentDf(dat)['Infected'].sum()
    total_infections = max( comp.drop(columns=['Time']).to_numpy().sum(), 0 )
    num_wt = comp[wt_seq].sum() if wt_seq in comp.columns else 0
    non_wt = total_infections - num_wt
    mut_frac.append( non_wt / max(total_infections,1) )
    inf_hosts.append(total_infected_hosts/num_hosts)

dat = df
dat['mutant_frac'] = mut_frac
dat['infected_hosts'] = inf_hosts
dat['Alive'] = dat['infected_hosts'] > 0

dat.to_csv('contact_sweep.csv')
dat = pd.read_csv('contact_sweep_mutfrec.csv')

# Plot mutant-only fraction of infected hosts
sns.set_style("ticks")
plt.figure(figsize=(8,2.4), dpi=300)
ax = sns.lineplot(
    x = 'contact_rate_host_host', y='mutant_frac', data=dat,
    color="#CC79A7", marker='o',ci=95
    )
ax.set_xlabel("Contact rate (1/unit time)")
ax.set_ylabel('Fraction of infected hosts \nwith only mutant pathogens')
plt.tight_layout()
plt.savefig('Fig2_replicate_mutfrec.png', bbox_inches='tight')

# Plot number of infected hosts (along with ODE prediction)
plt.figure(figsize=(8,2.4), dpi=300)
x = np.array(np.linspace(recovery_rate,2e-2,10000))
    # contact rates for ODE prediction
y = 1- (recovery_rate/x) # ODE prediction
ax = plt.plot(x, y, color="#ac6d2a", alpha=0.85) # plot ODE prediction
ax = sns.lineplot(
    x = 'contact_rate_host_host', y='infected_hosts', data=dat,
    color="#e69138", marker='o',ci=95,linestyle=''
    )
ax.set_xlabel("Contact rate (1/unit time)")
ax.set_ylabel('Fraction of hosts \ninfected at equilibrium')
plt.tight_layout()
plt.savefig('Fig2_replicate_infhos.png', bbox_inches='tight')
