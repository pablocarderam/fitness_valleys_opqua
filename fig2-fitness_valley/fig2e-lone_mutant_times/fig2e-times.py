
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

# Parameters
num_hosts = 1000
contact_rates = np.concatenate( # defines sweep
    [ np.linspace( 4e-3,1e-2,15 ), np.linspace( 1e-2,2e-2,10 ) ]
    )
print(contact_rates)
recovery_rate = 5e-3 # used later for plotting too

num_replicates = 1 # per contact rate
t_f = 10000 # final time point

# function to kill pathogens with resistant genome in this example
def lethalCheck(genome,lethal_genomes=[ re_seq ]):
    return 0 if genome in lethal_genomes else 1

def runReplicate(contact_rate,time_sampling):
    model = Model()
    model.newSetup( # Now, we'll define our new setup:
        'base_setup', preset='host-host', # Use default host-host parameters.
        possible_alleles='AB',
        num_loci=len(wt_seq),
        fitnessHost=fitnessLandscape,
        contactHost=lethalCheck,
        mutationHost=lethalCheck,
        mean_inoculum_host=1,
        contact_rate_host_host=contact_rate,
        recovery_rate_host=recovery_rate,
        mutate_in_host=2e-2,
        recombine_in_host=0
        )

    model.newPopulation('my_population','base_setup', num_hosts=num_hosts)
    model.addPathogensToHosts(
        'my_population',
        { wt_seq:int(num_hosts*0.5) }
            # start all simulations from a high number of infections
        )

    model.run(
        0,t_f,time_sampling=time_sampling, host_sampling=0, vector_sampling=0
        )

    return model

def saveModel(model,contact_rate):
    full_data = model.saveToDataFrame(
        'fitness_valley_beta='
            + str(np.around(contact_rate,decimals=5)) + '_rep=' + str(0) +'.csv'
        )

    full_data['Contact_rate'] = contact_rate
    full_data['Replicate'] = 0
    full_data.to_csv(
        'fitness_valley_beta='
            + str(np.around(contact_rate,decimals=5)) + '_rep=' + str(0) +'.csv'
        )

    return full_data

def processModel(full_data,contact_rate):
    all_times = []
    for pat_id in full_data['ID'].unique():
        pat = full_data.loc[full_data['ID']==pat_id,].fillna('')

        lone_mut_times = list(
            pat.loc[
                ~(pat['Pathogens'].str.contains(wt_seq, regex=False))
                & ~(pat['Pathogens']==''),'Time'
                ]
            )
        if len(lone_mut_times) > 0:
            non_lone_mut_times = list(
                pat.loc[
                ( pat['Time']>min(lone_mut_times) )
                & ( (pat['Pathogens'].str.contains(wt_seq, regex=False) )
                | ( pat['Pathogens']=='') ),'Time'
                ]
                )
            if len(non_lone_mut_times)==0:
                non_lone_mut_times = [lone_mut_times[-1]]

            lone_mut_times.append(-1)
            times = []
            start_idx = 0
            stop_idx = 0
            for i in range(len(lone_mut_times)):
                if ( lone_mut_times[i] >= non_lone_mut_times[stop_idx]
                        or lone_mut_times[i] < 0 ):
                    times.append(
                        non_lone_mut_times[stop_idx]-lone_mut_times[start_idx]
                        )
                    start_idx = i
                    if i < len(lone_mut_times)-1:
                        while ( stop_idx < len(non_lone_mut_times)-1
                                and non_lone_mut_times[stop_idx]
                                < lone_mut_times[start_idx]):
                            stop_idx += 1

            all_times = all_times + times

    return all_times


def runReplicates(contact_rates,time_sampling,n_cores=0):
    # If data already calculated, comment out from here...
    if not n_cores:
        n_cores = min(jl.cpu_count(),len(contact_rates))

    mod_arr = np.array( jl.Parallel(n_jobs=n_cores, verbose=1) (
        jl.delayed( runReplicate )
        (contact_rate,time_sampling) for contact_rate in contact_rates
        ) )

    full_datas = [
        saveModel(m,contact_rates[i]) for i,m in enumerate(mod_arr)
        ]
    # ...to here

    full_datas = [
        pd.read_csv('fitness_valley_beta='
            + str(np.around(contact_rate,decimals=5)) + '_rep=' + str(0) +'.csv'
            ) for i,contact_rate in enumerate(contact_rates)
        ]

    data_lists = jl.Parallel(n_jobs=n_cores, verbose=10) (
        jl.delayed( processModel )
        (d,contact_rates[i]) for i,d in enumerate(full_datas)
        )

    data_arrs = []
    for i in range(len(contact_rates)):
        data_arr = pd.DataFrame()
        data_arr['Times'] = data_lists[i]
        data_arr['Contact_rates'] = contact_rates[i]
        data_arrs.append(data_arr)

    out = pd.concat(data_arrs)# if len(data_arrs) > 0 else [0]

    return out

# If data already calculated, comment out from here...
dat = runReplicates(contact_rates,99)
dat.to_csv('lone_times.csv')
# ...to here
dat = pd.read_csv('lone_times.csv')


# Plots
# Plot mutant-only fraction of infected hosts
sns.set_style("ticks")
plt.figure(figsize=(8,2.4), dpi=300)
ax = sns.lineplot(
    x = 'Contact_rates', y='Times', data=dat,
    color="#0046ff", marker='o',ci=95
    )
ax.set_xlabel("Contact rate (1/unit time)")
ax.set_ylabel('Mean mutant survival time\n alone(time units)')
plt.tight_layout()
plt.savefig('Fig2_lone_times.png', bbox_inches='tight')
