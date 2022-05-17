
'''
Cardenas & Santos-Vega, 2021
Coded by github.com/pablocarderam
Creates plot of pathogen population composition at high transmission
used in Figure 2d
'''

import textdistance as td

import numpy as np
import matplotlib.pyplot as plt # plots
import seaborn as sns # pretty plots

from opqua.model import Model
from opqua.internal.data import compositionDf, compartmentDf

CB_PALETTE_mod = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0a2200",
                  "#0072B2", "#D55E00", "#f5d9ff", "#CC79A7", "#999999"]
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

modifier = 0.15 # modifies contact rate for the low transmission setting
beta = 1e-2 # base contact rate
num_hosts = 1000
inf_frac = 0.5 # base infected fraction
               # (this ultimately doesn't matter as long as it's high)

t_d = 10000 # time at which drug is applied

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
model.run(0,t_f,time_sampling=99, host_sampling=0, vector_sampling=0)
data = model.saveToDataFrame(
    'fitness_valley_example.csv'
    )

# Plot all genomes separately:
graph_compartments = model.compartmentPlot(
    'fitness_valley_example_compartments.png', data
    )

graph_composition = model.compositionPlot(
    'fitness_valley_example_composition_ind.png', data,
    num_top_sequences=6,
    count_individuals_based_on_model=model
    )

graph_clustermap = model.clustermap(
    'fitness_valley_example_clustermap.png', data,
    save_data_to_file='fitness_valley_example_pairwise_distances.csv',
    num_top_sequences=15,
    )

# Gropup genomes by number of B alleles:
for genome in model.global_trackers['genomes_seen']:
    data['Pathogens'] = data['Pathogens'].str.replace(
        genome,
        str( len(genome)-genome.count('B') ) + ' A, ' + str(genome.count('B')) + ' B'
        )

# Plot genomes grouped by number of B alleles:
comp_dat = compositionDf(
    data, track_specific_sequences=['8 A, 0 B','7 A, 1 B','6 A, 2 B','5 A, 3 B','4 A, 4 B','3 A, 5 B','2 A, 6 B','1 A, 7 B','0 A, 8 B'],
    num_top_sequences=-1,
    )

graph_composition = model.compositionPlot(
    'fitness_valley_example_composition.png', data,
    composition_dataframe=comp_dat,
    population_fraction=True,
    track_specific_sequences=['8 A, 0 B','7 A, 1 B','6 A, 2 B','5 A, 3 B','4 A, 4 B','3 A, 5 B','2 A, 6 B','1 A, 7 B','0 A, 8 B'],
    palette=CB_PALETTE_mod
    )
