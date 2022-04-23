
'''
Cardenas & Santos-Vega, 2022
Coded by github.com/pablocarderam
Creates plot of the combined host-vector fitness landscape used in Fig. S7
'''

import textdistance as td

import numpy as np
import matplotlib.pyplot as plt # plots
import seaborn as sns # pretty plots

from opqua.model import Model
from opqua.internal.data import compositionDf, compartmentDf

CB_PALETTE = ["#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]
    # www.cookbook-r.com/Graphs/Colors_(ggplot2)/#a-colorblind-friendly-palette
    # http://jfly.iam.u-tokyo.ac.jp/color/

# Defines intra-host competitive fitness of pathogens
def fitnessFunction(d,lam,x_shift):
    return ( 1-lam )**( np.abs( d-x_shift ) )

plt.rcParams.update({'font.size': 20})
sns.set_style("ticks")

fig = plt.figure(figsize=(10,6), dpi=300)
ax = fig.add_subplot(111)

d_series = np.linspace(-10,10,20+1)
plt.plot(
    d_series,[fitnessFunction(d,0.5,-3) for d in d_series],
    '-o', color=CB_PALETTE[0], linewidth=2,
    label="Pathogen fitness in host"
    )
plt.plot(
    d_series,[fitnessFunction(d,0.5, 3) for d in d_series],
    '-o', color=CB_PALETTE[1], linewidth=2,
    label="Pathogen fitness in vector"
    )

ax.set_xlabel("Genome distance")
ax.set_ylabel('Relative competitive fitness')
labels, locs = plt.xticks()
ax.set( xticklabels=np.abs( labels ) )
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand",ncol=3)
# plt.yscale('log')
plt.tight_layout()
plt.savefig('figS7-valley_length_landscapes.png', bbox_inches='tight')
