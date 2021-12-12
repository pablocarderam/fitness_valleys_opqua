'''
Cardenas & Santos-Vega, 2021
Coded by github.com/pablocarderam
Creates underlying plot of fitness landscape used in Figure 2a
The renderer is pretty glitchy, I had to touch up the plot afterwards :/
'''


import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# setup the figure and axes
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(121, projection='3d')

# Parameters
num_loci = 8
combinations = np.array([ math.comb( num_loci, k ) for k in range(num_loci+1) ])
num_comb = np.max(combinations)

# Fitness function used in simulations
def fitnessLandscape(genome):
    num_b = genome.count('B')
    if num_b < 3:
        f = 1 / ( 2 * ( (num_b+1)**5 ) )
    else:
        f = 1 / ( ( len(genome)-num_b+1 )**3 )

    return f

# vectors for bar X and Y location
x = np.arange(num_loci+1)
y = np.zeros_like(x)

# Z height of bars
top = np.array( [
    fitnessLandscape( ['B']*num_b + ['A']*(num_loci-num_b) )
    for num_b in range(num_loci+1)
    ] )[::-1]

top = np.log10( top*100+1 ) # log scale height
bottom = np.zeros_like(top) # bars start at 0
width = 1 # all bars have same width
depth = np.log10( combinations+1 ) # log scale depth

# 3D plot
ax1.bar3d(x, y, bottom, width, depth, top, shade=True, color='#ffcf65')

# Adjust axes
z_ticks = np.array([ 0, 0.01, 0.05, 0.125, 0.5, 1 ])
y_ticks = np.array([ 1, 10, 50, 100 ])
ax1.set_xticks( np.linspace(0.5,num_loci,num_loci) )
ax1.set_yticks( np.log10( y_ticks+1 ) )
ax1.set_zticks( np.log10( z_ticks*100+1 ) )

ax1.w_xaxis.set_ticklabels( np.linspace(1,num_loci,num_loci).astype(int)[::-1] )
ax1.w_yaxis.set_ticklabels( y_ticks )
ax1.w_zaxis.set_ticklabels( z_ticks )
ax1.set_xlabel('\nNumber of B alleles \nin genome')
ax1.set_ylabel('\nNumber of possible \ngenome combinations')
ax1.set_zlabel('\nRelative competitive \nfitness')

# will render into an interactive window
plt.show()
