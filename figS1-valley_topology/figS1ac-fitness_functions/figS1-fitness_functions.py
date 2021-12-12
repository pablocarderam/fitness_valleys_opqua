
'''
Cardenas & Santos-Vega, 2021
Coded by github.com/pablocarderam
Creates plot of contact rate sweep for survival used in Figure 2c
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

wt_seq = 'AAAAAAAA' # wild-type genome sequence
re_seq = 'BBBBBBBB' # drug resistant mutant genome sequence

# Defines intra-host competitive fitness of pathogens
def fitnessLandscape(num_b,num_alleles=8,frac_floor_raise=0):
    # num_b = genome[0:num_alleles].count('B')
    if num_b < min(3,num_alleles):
        f = ( 1 / ( (num_b+1)**5 ) * (1-frac_floor_raise) + frac_floor_raise )/2
    else:
        f = (1 / ( ( num_alleles-num_b+1 )**3 ) * (2-frac_floor_raise) + frac_floor_raise )/2

    return f

palette_length = CB_PALETTE[4:7]+[CB_PALETTE[0]]
palette_depth = CB_PALETTE[0:4][::-1]

plt.rcParams.update({'font.size': 20})
sns.set_style("ticks")

fig = plt.figure(figsize=(6,6), dpi=300)
ax = fig.add_subplot(111)
for i,num_allele in enumerate([2,4,6,8]):
    num_b_series = np.linspace(0,num_allele,num_allele+1)
    plt.plot(num_b_series,[fitnessLandscape(b,num_alleles=num_allele) for b in num_b_series],'-o', color=palette_length[i])

ax.set_xlabel("Number of B alleles \nin chosen loci")
ax.set_ylabel('Relative competitive fitness')
# ax.set_ylim(bottom=0, top=100)
# ax.set_xlim(left=-50, right=1050)
plt.yscale('log')
plt.tight_layout()
plt.savefig('figS1-valley_length_landscapes.png', bbox_inches='tight')

fig = plt.figure(figsize=(6,6), dpi=300)
ax = fig.add_subplot(111)
for i,rise in enumerate([0.05,0.1,0.2,0]):
    num_b_series = np.linspace(0,8,8+1)
    plt.plot(num_b_series,[fitnessLandscape(b,frac_floor_raise=rise) for b in num_b_series],'-o', color=palette_depth[i])

ax.set_xlabel("Number of B alleles \nin chosen loci")
ax.set_ylabel('Relative competitive fitness')
# ax.set_ylim(bottom=0, top=100)
# ax.set_xlim(left=-50, right=1050)
plt.yscale('log')
plt.tight_layout()
plt.savefig('figS1-valley_depth_landscapes.png', bbox_inches='tight')
