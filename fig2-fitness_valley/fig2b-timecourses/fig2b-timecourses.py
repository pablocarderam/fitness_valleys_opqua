
'''
Cardenas & Santos-Vega, 2021
Coded by github.com/pablocarderam
Creates plot of timecourses used in Figure 2b
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
beta = 1e-2 # base contact rate
num_hosts = 1000
inf_frac = 0.5 # base infected fraction
               # (this ultimately doesn't matter as long as it's high)

low_modifier = 0.15 # modifies contact rate for the low transmission setting
high_modifier = 3 # modifies contact rate for the high transmission setting

num_replicates = 100
t_d = 10000 # time at which drug is applied

def runReplicate(i,modifier=0.15, time_sampling=99):
    model = Model()
    model.newSetup( # Now, we'll define our new setup:
        'my_setup', preset='host-host', # Use default host-host parameters.
        possible_alleles='AB',
        num_loci=len(wt_seq),
        fitnessHost=fitnessLandscape,
        recoveryHost=drugCheck,
        mean_inoculum_host=1,
        contact_rate_host_host=(beta-5e-3) * modifier+5e-3,
        recovery_rate_host=5e-3,
        mutate_in_host=2e-2,
        recombine_in_host=0
        )

    model.newPopulation('my_population','my_setup', num_hosts=num_hosts)
    model.addPathogensToHosts(
        'my_population',
        { wt_seq : int(min(num_hosts*inf_frac*modifier,num_hosts*0.75)) }
        # initial number of infected hosts
        # this ultimately doesn't matter as long as it's high, it'll stabilize
        # if it's too low, you might either get extinction or an expansion
        # event that could increase diversity (and evolution of resistance)
        # in high transmission
        )

    model.newIntervention( t_d, 'treatHosts', [ 'my_population', 1, [re_seq] ] )
        # add drug, kills everything except resistant mutants

    t_f = t_d+5000 # final timepoint

    model.run(0,t_f,time_sampling=time_sampling, host_sampling=0, vector_sampling=0)

    return model


def processModel(model,i,modifier):
    full_data = model.saveToDataFrame(
        'fitness_valley_beta='
            + str(np.around((beta-5e-3) * modifier+5e-3,decimals=3)) + '_rep=' + str(i) + '.csv'
        )

    data = compartmentDf(full_data)
    data['Contact_rate'] = (beta-5e-3) * modifier+5e-3
    data['Replicate'] = i

    full_data['Contact_rate'] = (beta-5e-3) * modifier+5e-3
    full_data['Replicate'] = i
    full_data.to_csv(
        'fitness_valley_beta='
        + str(np.around((beta-5e-3) * modifier+5e-3,decimals=3)) + '_rep=' + str(i) + '.csv'
        )

    distances = np.array([ td.hamming( g,wt_seq ) for g in model.global_trackers['genomes_seen'] ])
    max_d = distances.max()
    var_d = np.var(distances)
    num_d = len(distances)

    return data


def runReplicates(num_replicates,modifier,time_sampling,n_cores=0):
    # If data already calculated, comment out from here...
    if not n_cores:
        n_cores = min(jl.cpu_count(),num_replicates)

    mod_arr = np.array( jl.Parallel(n_jobs=n_cores, verbose=1) (
        jl.delayed( runReplicate ) (i,modifier,time_sampling) for i in range(num_replicates)
        ) )

    data_arr = [ processModel(m,i,modifier) for i,m in enumerate(mod_arr) ]
    # ...to here

    data_arr = []
    for i in range(num_replicates):
        print('Loading replicate ' + str(i+1) + ' / ' + str(num_replicates))

        full_data = pd.read_csv(
            'fitness_valley_beta='
                + str(np.around((beta-5e-3) * modifier+5e-3,decimals=3)) + '_rep=' + str(i) + '.csv'
            )

        data = compartmentDf(full_data)
        data['Contact_rate'] = (beta-5e-3) * modifier+5e-3
        data['Replicate'] = i
        data_arr.append( data )

    return pd.concat(data_arr)

dat_high = runReplicates(num_replicates,high_modifier,299)
dat_low = runReplicates(num_replicates,low_modifier,99)


# Plots
# Plot timecourses
def plotLines(dat,color):
    plotted_survival = False
    for i in range(num_replicates):
        rep_dat = dat.loc[dat['Replicate']==i]

        # plot only one surviving time course in opaque color
        # (or last timecourse, and none survived)
        if ( rep_dat['Infected'].iloc[-1] > 0 or i==num_replicates ) \
                and not plotted_survival:
            plotted_survival = True
            ax.plot(
                rep_dat['Time'], rep_dat['Infected'], color=color, alpha=1
                )
        else:
            # plot the rest in transparent color
            ax.plot(
                rep_dat['Time'], rep_dat['Infected'],
                color=color, alpha=0.1, linewidth=0.75
                )

plt.figure(figsize=(8, 4), dpi=300)
ax = plt.subplot(1, 1, 1)

plotLines(dat_low,'#7ece5e')
plotLines(dat_high,'#ae5ece')

plt.xlabel('Time')
plt.ylabel('Infected hosts')

plt.savefig('Fig2_replicates.png', bbox_inches='tight')

# Make dataframe with final state of all replicates
low_arr = []
high_arr = []
for i in range(num_replicates):
    rep_dat_low = dat_low.loc[dat_low['Replicate']==i]
    rep_dat_high = dat_high.loc[dat_high['Replicate']==i]
    low_arr.append( rep_dat_low['Infected'].iloc[-1] )
    high_arr.append( rep_dat_high['Infected'].iloc[-1] )

dist_dat = pd.DataFrame()
dist_dat['Contact_rate'] = [(beta-5e-3) * low_modifier+5e-3]*num_replicates \
    + [(beta-5e-3) * high_modifier+5e-3]*num_replicates
dist_dat['Final_pop'] = low_arr + high_arr
dist_dat['Alive'] = dist_dat['Final_pop'] > 0

# Plot distribution of outcomes with kernel density
sns.set_style("ticks")
plt.figure(figsize=(2,4), dpi=300)
ax = plt.subplot(1, 1, 1)
ax = sns.histplot(
    y='Final_pop',hue='Contact_rate',data=dist_dat,
    palette=['#7ece5e','#ae5ece'], bins=10, kde=True,
    multiple="stack", log_scale=(False, False),
    stat='probability',legend= False
    )
ax.set_ylim(bottom=-40, top=840)
plt.savefig('Fig2_replicate_dist.png', bbox_inches='tight')

# Plot distribution of outcomes without kernel density
sns.set_style("ticks")
plt.figure(figsize=(2,4), dpi=300)
ax = plt.subplot(1, 1, 1)
ax = sns.histplot(
    y='Final_pop',hue='Contact_rate',data=dist_dat,
    palette=['#7ece5e','#ae5ece'], bins=10, kde=False,
    multiple="stack", log_scale=(False, False),
    stat='probability',legend= False
    )
ax.set_ylim(bottom=-40, top=840)
plt.savefig('Fig2_replicate_dist_nokde.png', bbox_inches='tight')
