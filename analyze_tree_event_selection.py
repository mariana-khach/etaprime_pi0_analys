#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:08:21 2022

@author: rupeshdotel
"""

import numpy as np
import LT.box as B
import matplotlib.pyplot as plt
import point_in_polygon_comp as pip

import root_util as RU



#%% lood root file result from ???? DSelector ???

#rfile = RU.root_tree('data/total_MC_small.root')
#rfile = RU.root_tree('data/flat_etaprpi0out_fall2018.root')
rfile = RU.root_tree('MC_signal_etaprimepi_small.root')
#rfile = RU.root_tree('data/data_phase1.root')


rfile.load_trees()
rfile.get_branches()

#%% store event data

d = rfile.tree_data

mpi0p = d['mpi0p']
metappi0 = d['metappi0']
metap = d['metap']

mpi013 = d['mpi013']
mpi024 = d['mpi024']
mpi014 = d['mpi014']
mpi023 = d['mpi023']

#cost_gj = d['cost_gj']
#phi_gj = d['phi_gj']

cost_hx = d['cost_hx']
phi_hx = d['phi_hx']


run_num = d['run_num'].astype(int)
event_num = d['event_num'].astype(int)

mpippimpi0 = d['mpippimpi0']

pi0_cost_gammapres = d['pi0_cost_gammapres']


px_pr = d['px_pr']
px_etapr = d['px_etapr']
px_pi0 = d['px_pi0']

py_pr = d['py_pr']
py_etapr = d['py_etapr']
py_pi0 = d['py_pi0']


pz_pr = d['pz_pr']
pz_etapr = d['pz_etapr']
pz_pi0 = d['pz_pi0']


e_pr = d['e_pr']
e_etapr = d['e_etapr']
e_pi0 = d['e_pi0']


px_beam = d['px_beam']
py_beam = d['py_beam']
pz_beam = d['pz_beam']
e_beam = d['e_beam']

pol = d['pol']

#%%

"""**************** 1. star shaped polygon cut 2pi0 ****************"""
#polygon vertex values for 2-pi-0 events
pv_x = np.array([.046, .112, .134, .161, 0.215, .166, .137, .092, .046])
pv_y = np.array([ .133, .161, .219, .168, .135, .113, .041, .104, .133 ])

# 2-pi0 events for 1-3 and 2-4 photon combinations
sel_two_pi0_13_24 = pip.polygon(pv_x, pv_y, mpi013, mpi024) > 0.

# 2-pi0 events for 1-4 and 2-3 photon combinations
sel_two_pi0_14_23 = pip.polygon(pv_x,  pv_y, mpi014, mpi023) > 0.

# select all possible 2-pi0 events
sel_poly_2pi0 =  sel_two_pi0_13_24 | sel_two_pi0_14_23

# select no 2 pi-0 events
no_2pi0 = (~sel_poly_2pi0) 

# histogram 1-3 and 2-4  2-gamma invariant masses
h2_2pi0_1324=B.histo2d(mpi013, mpi024, bins=(100,100),\
                       title=r"2 $\pi^0$ events for $(\gamma_1, \gamma_3)$ and $(\gamma_2, \gamma_4)$", \
                           xlabel = r'$M_{\gamma_{1},\gamma_{3}}$',\
                           ylabel = r'$M_{\gamma_{2},\gamma_{4}}$'    )

# histogram 1-4 and 2-3 2-gamma invariant masses    
h2_2pi0_1423=B.histo2d(mpi014, mpi023, bins=(100,100),\
                       title=r"2 $\pi^0$ events for $(\gamma_1, \gamma_4)$ and $(\gamma_2, \gamma_3)$", \
                           xlabel = r'$M_{\gamma_{1},\gamma_{4}}$',\
                           ylabel = r'$M_{\gamma_{2},\gamma_{3}}$'    )



# total number of evetns
tot_count = metap.shape[0]

# number of events w/o 2-pi0 events
sel_count = (~sel_poly_2pi0).sum()

# number and fraction of rejected events

rej_count = sel_poly_2pi0.sum()
rej_frac = rej_count/tot_count


print(70*'-')
print("For 2pi0 polygon cut")
print( 70*'-')
print(f'Total number of events = {tot_count:.0f}')
print(f'      selected  events counts = {sel_count:.0f}')
print(f'      rejected events counts = {rej_count:.0f}')
print(f'      rejected fraction of events = {rej_frac:.2f}')
print(70*'-')

#%% plotting and saveing the figures

B.pl.figure()
h2_2pi0_1324.plot(logz=True)
B.pl.plot(pv_x,pv_y,color="y",linewidth=2)
B.pl.savefig("Plots/2pi0_veto_1324.pdf")



B.pl.figure()
h2_2pi0_1423.plot(logz=True)
B.pl.plot(pv_x,pv_y,color="y",linewidth=2)
B.pl.savefig("Plots/2pi0_veto_1423.pdf")


#%%
      
"""**************** 2. Bayron rejection ****************"""
        


# mass and angle range for histogramming
lb_range=[[1,3.5],[-1,1]]    
  
h2_costhetapi0_Mpi0p=B.histo2d(mpi0p[no_2pi0], pi0_cost_gammapres[no_2pi0] , bins=(100,100), range=lb_range ,\
                               title=r'$\cos{\theta_{\pi^{0}}}$ vs $M_{\pi^{0},p}$', \
                                   xlabel = r'$M_{\pi^{0},p}$', \
                                       ylabel = r'$\cos{\theta_{\pi^{0}}}$', \
                                   )
# cut on polar angle of pi0
       
sel_backward_angle  = pi0_cost_gammapres < 0.0
sel_forward_angle = ~ sel_backward_angle

# resonance resgion: backward pi0
sel_Delta = no_2pi0 & sel_backward_angle
sel_no_2pi0_no_Delta = no_2pi0 & sel_forward_angle

# supress resonange region selectr forward pi0 

tot_count = no_2pi0.sum()
sel_count = sel_no_2pi0_no_Delta.sum()
rej_count = sel_Delta.sum()
rej_frac = rej_count/tot_count


h2_costhetapi0_Mpi0p_cut=B.histo2d(mpi0p[sel_Delta], pi0_cost_gammapres[sel_Delta] , bins=(100,100), range=lb_range ,\
                                   title="Rejected Resonances", \
                                       xlabel = r'$M_{\pi^{0},p}$', \
                                           ylabel = r'$\cos{\theta_{\pi^{0}}}$'\
                                               )

print(70*'-')
print("For baryon cut")
print( 70*'-')
print(f'Total number of events = {tot_count:.0f}')
print(f'      selected  events counts = {sel_count:.0f}')
print(f'      rejected events counts = {rej_count:.0f}')
print(f'      rejected fraction of events = {rej_frac:.2f}')
print(70*'-')

#%% Plotting of results

B.pl.figure()
h2_costhetapi0_Mpi0p.plot(logz=True)
B.pl.savefig("Plots/Costhetapi0_Mppi0.pdf")


B.pl.figure()
h2_costhetapi0_Mpi0p_cut.plot(logz=True)
B.pl.savefig("Plots/Costhetapi0_Mppi0_cut.pdf")

#%%
"""**************** 3. Omega rejection ****************"""


# histrogram ranges for omega studies
 
#axis_range=[[0.4,1.2],[0.93,1.0]]  
#original
axis_range=[[0.42,1.38],[0.9,1.02]]     
  
h2_Metaprime_Momega=B.histo2d(mpippimpi0[sel_no_2pi0_no_Delta], metap[sel_no_2pi0_no_Delta] , bins=(100,100), range=axis_range ,\
                              title="", \
                               xlabel = r'$M_{\pi^{+},\pi^{-},\pi^{0}}$', \
                                   ylabel = r'$M_{\pi^{+},\pi^{-},\eta}$'\
                                       )
# cut to select the omega    
#sel_omega = (0.73 < mpippimpi0) & (mpippimpi0 < 0.83)
sel_omega = (mpippimpi0 < -1)


# cut to avoid the omeha
sel_no_omega = ~sel_omega

sel_no_2pi0_no_Delta_no_omega = sel_no_omega & sel_no_2pi0_no_Delta

tot_count = sel_no_2pi0_no_Delta.sum()
sel_count = sel_no_2pi0_no_Delta_no_omega.sum()
rej_count = tot_count - sel_count
rej_frac = rej_count/tot_count


print(70*'-')
print("For omega cut")
print( 70*'-')
print(f'Total number of events = {tot_count:.0f}')
print(f'      selected  events counts = {sel_count:.0f}')
print(f'      rejected events counts = {rej_count:.0f}')
print(f'      rejected fraction of events = {rej_frac:.2f}')
print(70*'-')
       
h2_Metaprime_Momega_cut=B.histo2d(mpippimpi0[sel_no_2pi0_no_Delta_no_omega], \
                                  metap[sel_no_2pi0_no_Delta_no_omega] , bins=(100,50), range=axis_range ,\
                                      title="", \
                                          xlabel = r'$M_{\pi^{+},\pi^{-},\pi^{0}}$', \
                                              ylabel = r'$M_{\pi^{+},\pi^{-},\eta}$'\
                                                  )





#%% Plotting of results

B.pl.figure()
h2_Metaprime_Momega.plot()
B.pl.savefig("Plots/Metaprime_Momega.pdf")

B.pl.figure()
h2_Metaprime_Momega_cut.plot()       
B.pl.savefig("Plots/Metaprime_Momega_cut.pdf")
h2_Metaprime_Momega_cut.save(filename="Metaprime_Momega_omegaMC_rebin_origrange.data")


#%%
""" **************** 3. uninque event selection cut  **************** """

# form pairs of run_number and event_number
pair = np.array([run_num, event_num]).T #run number event number pair

# i_uniqe is index array of unique pairs into pair  
# pair_no_repeat are unique pairs of run number and event number
# pair_count are the number of occurences for each pair in pair_no_repeat in the original array

pair_no_repeat, i_unique,  pair_count = np.unique(pair, axis=0,  return_index = True, return_counts = True)

# only select those pairs that are unique : no multi photon cases
sel_pc= pair_count==1

# indices into the event array for the unique events
i_single_pairs = i_unique[sel_pc]

# make a selection array for the unique pairs
sel_unique_pairs = np.zeros_like(run_num).astype('bool')
sel_unique_pairs[i_single_pairs ] = True


# add the unique selection to the total cut
sel_all = sel_no_2pi0_no_Delta_no_omega & sel_unique_pairs


#frac of events rejected
tot_count = sel_no_2pi0_no_Delta_no_omega.sum()               # total number of good events 
sel_count = sel_all.sum() # total number of unique events
rej_count = tot_count - sel_count  
rej_frac = rej_count/tot_count


print(70*'-')
print("For uniqe event selection")
print( 70*'-')
print(f'Total number of events = {tot_count:.0f}')
print(f'      selected  events counts = {sel_count:.0f}')
print(f'      rejected events counts = {rej_count:.0f}')
print(f'      rejected fraction of events = {rej_frac:.2f}')
print(70*'-')



#%% save currently selected events

RU.save_dict('signal_small.npz', rfile.tree_data, selection = sel_all)



