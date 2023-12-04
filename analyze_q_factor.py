#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:08:21 2022


perform a q-factor analysis for good, unique events

@author: rupeshdotel

modified: WB

"""

import numpy as np
import LT.box as B
import matplotlib.pyplot as plt

import scipy.spatial as SP

# local modules (just python files)
import root_util as RU
import prep_PWA as PWA

dtr = np.pi/180.

#%% q-factor analysis


"""***************** 4. analysis for  q-factors ************"""

# setup fit function for q-factor
#q-factors

# inital values

# gaussian fit parameters with initial values
A = B.Parameter(50., 'A')
x0 = B.Parameter(0.956, 'x0')
sigma = B.Parameter(.005, 'sigma')


# linear background
a0 = B.Parameter(0., 'a0')
a1 = B.Parameter(0., 'a1')

def gaus(x):
    return A()*np.exp(-(x - x0() )**2/(2.*sigma()**2))

def lin_bkg(x):
    return a0() + a1()*x 

def signal(x):
    return gaus(x) + lin_bkg(x)


#%% load the data file

#event_file = '2017_widecoherent_good_events.npz'
#event_file = 'MC_rec_fall2018.npz'
event_file='signal_small.npz'


print(f'Loading event file {event_file}')

ed = RU.read_npz(event_file)


# get the needed data from the event selection

cost_hx = ed['cost_hx']
phi_hx = ed['phi_hx'] * dtr


M_inv = ed['metap']


pol = ed['pol']

px_pr = ed['px_pr']
px_etapr = ed['px_etapr']
px_pi0 = ed['px_pi0']

py_pr = ed['py_pr']
py_etapr = ed['py_etapr']
py_pi0 = ed['py_pi0']

pz_pr = ed['pz_pr']
pz_etapr = ed['pz_etapr']
pz_pi0 = ed['pz_pi0']

e_pr = ed['e_pr']
e_etapr = ed['e_etapr']
e_pi0 = ed['e_pi0']

px_beam = ed['px_beam']
py_beam = ed['py_beam']
pz_beam = ed['pz_beam']
e_beam = ed['e_beam']

"""

#%%
cos_theta_hx = cost_hx

# phi helicity
cos_phi_hx = np.cos(phi_hx) 
sin_phi_hx = np.sin(phi_hx) 

# pairs od cos(phi) and sin(phi) values
cosphi_sinphi_hx_pair = np.array([cos_phi_hx ,sin_phi_hx]).T 


# pairs of
cos_theta_hx_pair = np.array([cos_theta_hx,cos_theta_hx]).T 

# calculate the distance between all gj angles in array 
dcos_theta_hx_a = (SP.distance.pdist(cos_theta_hx_pair))/2. #divide by 2 to correct for double counting

#calculate the distance between all gj phi angles in array 
dphi_hx_a = SP.distance.pdist(cosphi_sinphi_hx_pair)

# cos_theta_gj_range is the total range of costheta variable
cos_theta_hx_range = 4.0

# phi_gj_range is the maximum distance possible between (cosphi, sinphi) points 
phi_hx_range = 8.0


#%%
# convert distance array  to symmetric matrix and normalize
dcos_theta_hx_m = SP.distance.squareform(dcos_theta_hx_a)/cos_theta_hx_range

#%%
#takes a while about 1 minute
dphi_hx_m = SP.distance.squareform(dphi_hx_a)/phi_hx_range


#%%


# Nf are  the number of neighboring events we choose
n_near = 400


qf = np.zeros_like(M_inv)
q_err = np.zeros_like(M_inv)

cl = np.zeros_like(M_inv)
chi2_red = np.zeros_like(M_inv)


# X0 are the mean values from the fit
X0 = np.zeros_like(M_inv)

# S0 are the sigma values from the fit
S0 = np.zeros_like(M_inv)


#%% loop over each event to determine q-factor, use the eta' invariant mass as discriminating variable

# select if each fit should be plotted
do_plot  = False


for i,M_inv_loc in enumerate(M_inv[:]):
    
    # M_inv_loc isw the M_inv value of the current event
    
    # select neghboring events for the current event combine distance of costheta and phi normalized by  pythagorian theorem   
    i_near = np.argsort(np.sqrt((dcos_theta_hx_m[i]**2 +  dphi_hx_m[i]**2 )))[:n_near] 
                                    
    # select the invariant masses for the nearest neighbors
    M_inv_neighbor = M_inv[i_near]
   
    # make a histgram with the nearest neighbors
    h = B.histo(M_inv_neighbor, bins = 20)
    # 
    h_sel = h.bin_content > 0
    M = h.bin_center[h_sel]
    C = h.bin_content[h_sel]
    dC = h.bin_error[h_sel]
    
    # fit a gaussian on a linear background to the data
    fit = B.genfit(signal, [A, x0, sigma,  a0, a1],
                   x = M, y = C, y_err = dC, print_results = False, plot_fit = False )
    
    if do_plot:
        plt.figure()
        B.plot_exp(M,C,dC)
        B.plot_line(fit.xpl, fit.ypl, color = 'r')
        mr = np.linspace(M[0], M[-1], 1000)
        plt.xlabel(r"$M(\pi^{+}\pi^{-}\eta) GeV/c^{2}$", fontsize = 16)
        plt.title(f"Inv mass bin {i}")
        B.plot_line(mr, gaus(mr), color = 'm')
        B.plot_line(mr, lin_bkg(mr), color = 'g')
        # event location indicated by a black line
        plt.vlines(M_inv_loc, h.bin_content.min(),  h.bin_content.max(), color = 'black')
    
    # determin the q-factor
    ws = gaus(M_inv_loc)   # signal value 
    wb = lin_bkg(M_inv_loc) # background value
    # calculate q-factor  
    q = ws/(ws+wb)
    # store the value
    qf[i] = q
    # store fit parameters
    X0[i] = x0.value   # position
    S0[i] = sigma.value # sigma
    
    cl[i] = fit.CL  # confidence level
    chi2_red[i] = fit.chi2_red # reduced chi dq.

#%% select valid q-factors

sel_qf = (qf > 0) & (qf < 1)  # q-value must be between 0 and 1

qsel = qf[sel_qf]   # these are the selected signal values
    
"""

qf = (np.ones_like(M_inv))
sel_qf = (np.ones_like(M_inv)).astype(dtype=bool)
qsel = qf[sel_qf] 

#%%

"""*************** 5. separate polarization**************"""


pol_f = pol[sel_qf]

px_pr_f = px_pr[sel_qf]
px_etapr_f = px_etapr[sel_qf]
px_pi0_f = px_pi0[sel_qf]

py_pr_f = py_pr[sel_qf]
py_etapr_f = py_etapr[sel_qf]
py_pi0_f = py_pi0[sel_qf]

pz_pr_f = pz_pr[sel_qf]
pz_etapr_f = pz_etapr[sel_qf]
pz_pi0_f = pz_pi0[sel_qf]

e_pr_f = e_pr[sel_qf]
e_etapr_f = e_etapr[sel_qf]
e_pi0_f = e_pi0[sel_qf]

px_beam_f = px_beam[sel_qf]
py_beam_f = py_beam[sel_qf]
pz_beam_f = pz_beam[sel_qf]
e_beam_f = e_beam[sel_qf]





#%%
# polarization selections

sel_amo = pol_f == -1
sel_0 = pol_f == 0
sel_90 = pol_f == 90
sel_45 = pol_f == 45
sel_135 = pol_f == 135

# the selected polarization

pol_values = ['0','45','90','135']


pol_dict = {'0':sel_0, '45':sel_45, '90':sel_90, '135':sel_135, 'amo':sel_amo}


# sel_pol is the polarization angle, can be 0, 45, 90 135
# possible vlaues sel_amo | sel_0 | sel_90 | sel_45 | sel_135


loop_pol=[key for key in pol_dict.keys()]


#Loop through different polarization orientations and save corresponding  "kin" tree to a .root file
for polarization in loop_pol[:4]:

    
      # select the correponding values
    sel_pol = pol_dict[polarization]
      
      # selected polarizztions
    pol_s = pol_f[sel_pol]
      
    px_pr_s = px_pr_f[sel_pol]
    px_etapr_s = px_etapr_f[sel_pol]
    px_pi0_s = px_pi0_f[sel_pol]
    py_pr_s = py_pr_f[sel_pol]
    py_etapr_s = py_etapr_f[sel_pol]
    py_pi0_s = py_pi0_f[sel_pol]
      
    pz_pr_s = pz_pr_f[sel_pol]
    pz_etapr_s = pz_etapr_f[sel_pol]
    pz_pi0_s = pz_pi0_f[sel_pol]
      
    e_pr_s = e_pr_f[sel_pol]
    e_etapr_s = e_etapr_f[sel_pol]
    e_pi0_s = e_pi0_f[sel_pol]
      
    px_beam_s = px_beam_f[sel_pol]
    py_beam_s = py_beam_f[sel_pol]
    pz_beam_s = pz_beam_f[sel_pol]
    e_beam_s = e_beam_f[sel_pol]
      
    qf_s = qsel[sel_pol]
      
      
      #%%
      # make output dictionary
      
    d_pwa = {}
      
    d_pwa['px_pr'] = px_pr_s
    d_pwa['px_etapr'] = px_etapr_s
    d_pwa['px_pi0'] = px_pi0_s
      
      
    d_pwa['py_pr'] = py_pr_s
    d_pwa['py_etapr'] = py_etapr_s
    d_pwa['py_pi0'] = py_pi0_s
      
      
    d_pwa['pz_pr'] = pz_pr_s
    d_pwa['pz_etapr'] = pz_etapr_s
    d_pwa['pz_pi0'] = pz_pi0_s
      
    d_pwa['e_pr'] = e_pr_s
    d_pwa['e_etapr'] = e_etapr_s
    d_pwa['e_pi0'] = e_pi0_s
      
    d_pwa['px_beam'] = px_beam_s
    d_pwa['py_beam'] = py_beam_s
    d_pwa['pz_beam'] = pz_beam_s
    d_pwa['e_beam'] = e_beam_s
      
    d_pwa['pol'] = pol_s
    d_pwa['qf'] = qf_s
      
      
      #%% save the data to file
    #  RU.save_dict(f"data/pwa/2017_wideEpeak_pol_{polarization}.npz", d_pwa)       
      
      #%% or you could also make the kin tree directly here
      
    PWA.make_kin_tree(d_pwa, f"data/pwa/kin_tree_{polarization}.root")




             





