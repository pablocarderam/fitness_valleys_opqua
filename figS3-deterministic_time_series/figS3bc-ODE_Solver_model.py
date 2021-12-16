#!/usr/bin/env python3

'''
Deterministic numerical solver for ODE systems
Pablo Cardenas R.

used for
Cardenas & Santos-Vega, 2021
Coded by github.com/pablocarderam
Creates heatmaps of contact rate and mutant fitness cost used in Figure 3b-c
'''


### Imports ###
import numpy as np # handle arrays
import pandas as pd
from scipy import integrate # numerical integration
import joblib as jl
import itertools as it

import seaborn as sns # for plots
import matplotlib.pyplot as plt

sns.set_style("white") # make pwetty plots
cb_palette = ["#999999", "#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    # http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/#a-colorblind-friendly-palette
    # http://jfly.iam.u-tokyo.ac.jp/color/



### Methods ###
# User-defined methods #
def params():

    """
    Returns default values constant values for the model in a dictionary.
    """

    params = {
        't_0':0,        # time - Initial time value
        't_f':1e3,      # time - Final time value
        't_den':1,      # time - Size of time step to evaluate with

        'd':1e-1,       # 1/time - host recovery rate
        'b':1.2e-1,     # 1/time - host contact rate
        'f':1/(2-0.5),  # no units - probability of WT outcompeting mutants in
                        #            intra-host competition
        'n':1,          # pathogens - size of inoculum
        'm_1':5e-2,     # 1/time - mutation rate from WT to mutant
        'm_2':0,        # 1/time - mutation rate from mutant to WT
        'N':500,        # hosts - total population size
    }

    return params


def initCond():

    '''
    Return initial conditions values for the model in a dictionary.
    '''

    y0 = [
        # Initial concentrations in [M] (order of state variables matters)
        params()['N']-10,   # hosts - susceptible, uninfected hosts S
        10,                 # hosts - WT infected hosts W
        0,                  # hosts - mutant infected hosts M
        0,                  # hosts - WT and mutant coinfected hosts C
        ]

    return y0


def odeFun(t,y,**kwargs):

    """
    Contains system of differential equations.

    Arguments:
        t        : current time variable value
        y        : current state variable values (order matters)
        **kwargs : constant parameter values, interpolanting functions, etc.
    Returns:
        Dictionary containing dY/dt for the given state and parameter values
    """

    S,W,M,C = y # unpack state variables
    # (state variable order matters for numerical solver)

    # Unpack variables passed through kwargs (see I thought I could avoid this
    # and I just made it messier)
    d,b,f,n,m_1,m_2,N = \
        kwargs['d'],kwargs['b'],kwargs['f'],kwargs['n'], \
        kwargs['m_1'],kwargs['m_2'],kwargs['N']

    # ODEs
    dS = ( d - b * S / N ) * ( N - S )
    dW = ( b / N ) * ( W * ( S - M - ( 1 - np.power(f,n) ) * C ) + np.power(f,n) * S*C ) - ( d+m_1 ) * W
    dM = ( b / N ) * ( M * ( S - W - ( 1 - np.power(1-f,n) ) * C ) + ( np.power(1-f,n) ) * S*C ) - ( d+m_2 ) * M
    dC = ( b / N ) * ( C * ( S * ( 1-np.power(f,n)-np.power(1-f,n) ) + ( 1 - np.power(f,n) ) * W + ( 1 - np.power(1-f,n) ) * M ) + 2 * W * M ) + m_1 * W + m_2 * M - d * C

    # Gather differential values in list (state variable order matters for
    # numerical solver)
    dy = [dS,dW,dM,dC]

    return dy


def figTSeries(sol,f_name='ODE_tseries.png'):

    """
    This function makes a plot for Figure 1 by taking all the solution objects
    as arguments, and prints out the plot to a file.
    Arguments:
        sol : solution object taken from solver output
    """

    t = sol.t[:] # get time values

    plt.figure(figsize=(6, 4), dpi=200) # make new figure

    ax = plt.subplot(1, 1, 1) # Fig A
    plt.plot(t, sol.y[0,:], label=r'$S$', color=cb_palette[2])
    plt.plot(t, sol.y[1,:], label=r'$W$', color=cb_palette[4])
    plt.plot(t, sol.y[2,:], label=r'$M$', color=cb_palette[6])
    plt.plot(t, sol.y[3,:], label=r'$C$', color=cb_palette[1])
    plt.xlabel('Time (h)')
    plt.ylabel('Hosts')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)

    plt.savefig(f_name, bbox_inches='tight')

# Pre-defined methods #
# These shouldn't have to be modified for different models
def odeSolver(func,t,y0,p,solver='LSODA',rtol=1e-8,atol=1e-8,**kwargs):

    """
    Numerically solves ODE system.

    Arguments:
        func     : function with system ODEs
        t        : array with time span over which to solve
        y0       : array with initial state variables
        p        : dictionary with system constant values
        solver   : algorithm used for numerical integration of system ('LSODA'
                   is a good default, use 'Radau' for very stiff problems)
        rtol     : relative tolerance of solver (1e-8)
        atol     : absolute tolerance of solver (1e-8)
        **kwargs : additional parameters to be used by ODE function (i.e.,
                   interpolation)
    Outputs:
        y : array with state value variables for every element in t
    """

    # default settings for the solver
    options = { 'RelTol':10.**-8,'AbsTol':10.**-8 }

    # takes any keyword arguments, and updates options
    options.update(kwargs)

    # runs scipy's new ode solver
    y = integrate.solve_ivp(
            lambda t_var,y: func(t_var,y,**p,**kwargs), # use a lambda function
                # to sub in all parameters using the ** double indexing operator
                # for dictionaries, pass any additional arguments through
                # **kwargs as well
            [t[0],t[-1]], # initial and final time values
            y0, # initial conditions
            method=solver, # solver method
            t_eval=t, # time point vector at which to evaluate
            rtol=rtol, # relative tolerance value
            atol=atol # absolute tolerance value
        )

    return y

# Solving model
# To generate data, uncomment the following...
# Single timecourse
def solveModel():

    '''
    Main method containing single solver and plotter calls.
    Writes figures to file.
    '''

    # Set up model conditions
    p = params() # get parameter values, store in dictionary p
    y_0 = initCond() # get initial conditions
    t = np.linspace(p['t_0'],p['t_f'],int(p['t_f']-p['t_0']/p['t_den']) + 1)
        # time vector based on minimum, maximum, and time step values

    # Solve model
    sol = odeSolver(odeFun,t,y_0,p,solver="Radau");

    # Call plotting of figure 1
    figTSeries(sol,f_name='ODE_tseriesL-low_b.png')

    # Set up model conditions
    p = params() # get parameter values, store in dictionary p
    p['b'] = 2e-1
    y_0 = initCond() # get initial conditions
    t = np.linspace(p['t_0'],p['t_f'],int(p['t_f']-p['t_0']/p['t_den']) + 1)
        # time vector based on minimum, maximum, and time step values

    # Solve model
    sol = odeSolver(odeFun,t,y_0,p,solver="Radau");

    # Call plotting of figure 2
    figTSeries(sol,f_name='ODE_tseriesL-high_b.png')

    plt.close()

solveModel()
