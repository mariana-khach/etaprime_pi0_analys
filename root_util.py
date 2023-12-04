#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:34:47 2022

Utility functions for root trees based on uproot

not root_numpy is no longer supported.

@author: boeglinw
"""

import numpy as np
import uproot as UR



#%%
def save_dict(filename, dictionary, selection = None):
    """
    Save a dictionary to an npz file based on a selection 

    Parameters
    ----------
    filename : string
        filename for the output file.
    dictionary: dict
        dictionary to save
    selection : list of bools, optional
        bool arrays for the selected lines. The default is None, meaning all data.

    Returns
    -------
    None.

    """
    d = dictionary
    arg_names = list(d.keys())
    if selection is None:
        np.savez(filename, **d)
    else:
        # apply selection to all data, make temporary dictionary
        dd = dict(zip(list(d.keys()),[d[k][selection] for k in d.keys()]))
        np.savez(filename, **dd)


def save_dict_to_root(filename, dictionary, tree_name = "dict_tree", selection = None):
    """
    Save a dictionary as a root tree based on a selection. It is important
    that all key data have the same length

    Parameters
    ----------
    filename : string
        filename for the output file.
    dictionary: dict
        dictionary to save
    selection : list of bools, optional
        bool arrays for the selected lines. The default is None, meaning all data.

    Returns
    -------
    None.

    """
    d = dictionary
    arg_names = list(d.keys())
    if selection is None:
        r_file = UR.recreate(filename)
        r_file[tree_name] = d
        r_file.close()
    else:
        # apply selection to all data, make temporary dictionary
        dd = dict(zip(list(d.keys()),[d[k][selection] for k in d.keys()]))
        r_file = UR.recreate(filename)
        r_file[tree_name] = dd
        r_file.close()        
        
class read_npz:
    
    def __init__(self, filename):
        self.filename = filename
        self.data = np.load(filename)
        keys = list(self.data.keys())
        self.keys = keys
        
    def __getitem__(self, x):
        return self.data[x]
    

        

#%%
class root_tree: 
    
    def __init__(self, file_name, trees = ['etaprpi0_Tree']):
        """
        read a root file containing trees

        Parameters
        ----------
        file_name : string 
            root file name
        trees : list of strings
            Tree names the sould be loaded The default is ['etaprpi0_Tree'].

        Returns
        -------
        None.

        """
        self.filename = file_name
        self.tree_names = trees
        self.root_file = UR.open(self.filename)
        
    def load_trees(self):
        """
        Load all requested trees and store them in a dictionary called
        tree_names. The keys are the tree names

        Returns
        -------
        None.

        """
        self.trees = {}
        for t_name in self.tree_names:
            try :
                self.trees[t_name] = self.root_file[t_name]
            except Exception as err:
                print(f'cannot get tree {t_name}: {err}')
                continue
        # select the first tree by default
        if self.trees != {}:
            k0 = list(self.trees.keys())[0]
            self.selected_tree = self.trees[k0]
            
    def select_tree (self, name):
        """
        select a tree from theas the current working tree. The default tree is the 
        first one in the list.

        Parameters
        ----------
        name : string
            tree name

        Returns
        -------
        None.

        """
        try:
            self.selected_tree = self.trees[name]
        except Exception as err:
            print(f'cannot select tree {name}: {err}')

    def list_branches(self, tree_name = ''):
        """
        list all branches in tree tree_name
        if no tree_name use the selected one        

        Parameters
        ----------
        tree_name : string, optional
            the name of the tree. The default is ''.

        Returns
        -------
        None.

        """

        if tree_name == '':
            tree = self.selected_tree
        else:
            try:
                tree = self.trees[tree_name]
            except Exception as err:
                print(f'cannot get tree {tree_name}: {err}')
                return
        for k in tree.keys():
            print(k)
        

    def get_branches(self, branch_list = []):
        """
        
        load branch data from tree and store them either in a dictionary

        
         
         If the branch list is an empty list all branches are loaded
         
        Parameters
        ----------
        branch_list :  list of string, optional
            list of branch names to be loaded. The default is [] (all branches are loaded.
        Returns
        -------
        None.

        """
        
        tree = self.selected_tree
        if branch_list == []:
            tree_data = tree.arrays(library = "np")
        else:
            tree_data = tree.arrays(branch_list, library = "np")
        self.tree_data = tree_data
            

    def delete_branches(self, branch_list):
        """
        remove the list of tree branch data. This is done if you do not
        need the data anymore to free up memory        

        Parameters
        ----------
        branch_list : list of strings
            list of branches to be removed.

        Returns
        -------
        None.

        """

        for b in branch_list:
            self.tree_data.pop(b)
                

        