#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:32:17 2021

@author: rupeshdotel

generate kin tree for PWA from clean samples in the form of numpy arrays

"""

import numpy as np

import root_util as RU

#import ROOT as R



#%%

def npz_to_kintree(f):

    # final state vectors for 3 final particles
    # x-components
    px_pr = f['px_pr']        # proton  
    px_etapr = f['px_etapr']  # etaprime
    px_pi0 = f['px_pi0']      # pi0

    # y-components
    py_pr = f['py_pr']
    py_etapr = f['py_etapr']
    py_pi0 = f['py_pi0']

    # z-components
    pz_pr = f['pz_pr']
    pz_etapr = f['pz_etapr']
    pz_pi0 = f['pz_pi0']
    
    # total energies
    e_pr = f['e_pr']
    e_etapr = f['e_etapr']
    e_pi0 = f['e_pi0']

    # beam vector
    px_beam = f ['px_beam']
    py_beam = f['py_beam']
    pz_beam = f['pz_beam']
    e_beam = f['e_beam']


    # setup arrays, this should be mad automatic so that it corresponds to the AmpTools config file.
    
    Px = np.array([px_pr, px_etapr, px_pi0]).T
    Py = np.array([py_pr, py_etapr, py_pi0]).T
    Pz = np.array([pz_pr, pz_etapr, pz_pi0]).T
    E = np.array([e_pr, e_etapr, e_pi0]).T
    
    return E, Px, Py, Pz, e_beam, px_beam, py_beam,  pz_beam


#%%


def make_kin_tree(d , output_filename = 'kin_tree.root'):
    # is a dictionary as produced in analyze_q_factor
    # prepare the data as required in the AmpTools analysis    
    E, Px, Py, Pz, e_beam, px_beam, py_beam,  pz_beam = npz_to_kintree(d)
    
    # prepare the tree dictionary
    d_root = {}
    
    d_root['E_FinalState'] = E.astype(np.float32)
    d_root['Px_FinalState'] = Px.astype(np.float32)
    d_root['Py_FinalState'] = Py.astype(np.float32)
    d_root['Pz_FinalState'] = Pz.astype(np.float32)
    
    d_root['E_Beam'] = e_beam.astype(np.float32)
    d_root['Px_Beam'] = px_beam.astype(np.float32)
    d_root['Py_Beam'] = py_beam.astype(np.float32)
    d_root['Pz_Beam'] = pz_beam.astype(np.float32)
    
    # number of final state particles (always 3 in this case)
    num_fs = np.repeat(3, e_beam.shape[0])
    d_root['NumFinalState'] = num_fs.astype(np.int32)
    
    # polarization
    d_root['Polarization_Angle'] = d['pol'].astype(np.float32)    
    
    RU.save_dict_to_root(output_filename, d_root, tree_name = 'kin')


def make_kin_tree_from_file(input_file, output_file = 'kin_tree.root'):
    # first load the saved data
    d_loc = RU.read_npz(input_file)
    make_kin_tree(d_loc, output_filename = output_file)
    
#%% example make a kin tree from the file previously created

make_kin_tree_from_file('./data/pwa/2017_wideEpeak_pol_135.npz',
                        output_file = './data/pwa/AmpTool_2017_pol_135.root')

