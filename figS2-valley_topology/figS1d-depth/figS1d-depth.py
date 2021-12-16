
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

# Defines intra-host competitive fitness of pathogens in terms of the length of
# the fitness valley (number of alleles that affect fitness) and the fraction of
# the original fitness drop that is added back (0 means original drop, 1 means
# there is no drop relative to WT fitness)
def fitnessLandscape(genome,num_alleles=8,frac_floor_raise=0):
    num_b = genome[0:num_alleles].count('B')
    if num_b < 3:
        f = ( 1 / ( (num_b+1)**5 ) * (1-frac_floor_raise) + frac_floor_raise )/2
    else:
        f = (1 / ( ( num_alleles-num_b+1 )**3 ) * (2-frac_floor_raise) + frac_floor_raise )/2

    return f

# Modifies recovery rate of resistant pathogens, e.g. if hosts are seeking
# treatment by a drug when they get sick and the resistant pathogen endures for
# longer.
# Included in example as a way of increasing mutant persistance, makes effect
# easier to visualize.
def drugCheck(genome,re_seq='BBBBBBBB',num_alleles=8):
    return 1 if genome[0:num_alleles]!=re_seq[0:num_alleles] else 0.75

# Parameters
num_hosts = 1000
contact_rates = np.concatenate( # defines sweep
    [ np.linspace( 2e-2,4e-3,12 ) ]
    )

num_replicates = 100 # per contact rate
t_d = 10000 # time at which drug is applied
t_f = t_d+500 # final time

# This function stops the simulation early if there are more than a certain
# number of resistant genotypes in the model state
def stopSimIfResistant(
        model,sample_size=100,res_frac=0.1,re_seq='BBBBBBBB',num_alleles=8,
        t_f=t_f):
    num_r = 0 # counts hosts with resistant pathogens
    count = 0 # total infected hosts counted
    for h in model.populations['my_population'].hosts:
            # loop over all hosts
        if len(h.pathogens) > :
            count += 1 # count this host
            if re_seq[0:num_alleles] in [p[0:num_alleles] for p in h.pathogens]:
                    # if resistant genotypes are in this host's pathogens,
                num_r += 1 # count as resistant infection
                if num_r > res_frac*sample_size:
                        # if fraction of resistant infections exceeds threshold,
                    print('Killswitch: ' + str(model.t_var) + str(count))
                    model.t_var = t_f+1 # skip time to end
                    break # stop counting

            elif count > sample_size:
                break # else if done counting hosts for sample, stop

# This function runs a set of simulations with the given fitness landscape
# function
def runFitness(fitnessLandscape,drugCheck,stopSimIfResistant,checkpoints=10):
    print('*** RUNNING BATCH ***')

    model = Model()

    for t in np.linspace( 0,t_d,checkpoints,endpoint=False )+(t_d/checkpoints):
        model.newIntervention( t, 'customModelFunction', [stopSimIfResistant] )

    model.newIntervention( t_d+1, 'customModelFunction', [stopSimIfResistant] )

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

    return dat

# Uncomment the following to run simulations:
dat_0 = runFitness(
    lambda g: fitnessLandscape(g,frac_floor_raise=0),
    drugCheck,
    stopSimIfResistant
    )

dat_1 = runFitness(
    lambda g: fitnessLandscape(g,frac_floor_raise=0.05),
    drugCheck,
    stopSimIfResistant
    )

dat_2 = runFitness(
    lambda g: fitnessLandscape(g,frac_floor_raise=0.1),
    drugCheck,
    stopSimIfResistant
    )

dat_3 = runFitness(
    lambda g: fitnessLandscape(g,frac_floor_raise=0.2),
    drugCheck,
    stopSimIfResistant
    )

dat_0['depth'] = '1.00 original drop'
dat_1['depth'] = '0.95 original drop'
dat_2['depth'] = '0.90 original drop'
dat_3['depth'] = '0.80 original drop'

dat = pd.concat([dat_0,dat_1,dat_2,dat_3])
dat.to_csv('contact_sweep_survival_depth_full.csv')
# ...until here

dat = pd.concat([
    pd.read_csv('contact_sweep_survival_depth_full.csv')
    ])

# Plot
sns.set_style("ticks")
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(8,4), dpi=300)
ax = sns.lineplot(
    x = 'contact_rate_host_host', y='Alive', hue='depth', data=dat,
    palette=CB_PALETTE[0:4][::-1], marker='o',ci=95
    )
ax.set_xlabel("Contact rate (1/unit time)")
ax.set_ylabel(
    'Probability of resistant \npathogens surviving treatment')
plt.tight_layout()
ax.get_legend().remove()
plt.savefig('FigS1-valley_depth.png', bbox_inches='tight')
