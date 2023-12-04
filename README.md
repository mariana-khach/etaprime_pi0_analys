# etaprime_pi0_analys
In this directory we have to python codes that are used to apply final event selection on etaprime pi0 data, after initial selection applied in C++ based DSelector analysis codes.
The output of DSelector code is a ROOT tree containing necessay info for each event, which analyze_tree_event_selection.py takes as an input and
applies the rest of the cuts and saves the output in an .npz file.
The .npz file creaed in the step above is then processed through analyze_q_factor.py to assign signal weights to each of the events based on Q-factor technique.
