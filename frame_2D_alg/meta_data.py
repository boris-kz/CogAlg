from copy import copy, deepcopy
import numpy as np

ave,avd = .3,.5; decay = ave/(ave+avd)  # ave m,d / unit dist, recomputed from dTT*wTT?
wY, wX = 64, 64; wYX = np.hypot(wY,wX)

eps = 1e-7
wM,wD,wi, wG,wI,wa, wL,wS,wA = 10, 10, 20, 20, 5, 20, 2, 1, 1  # dTT weights = reversed relative ave, update from wTT_ after feedback
wT = np.array([wM,wD,wi, wG,wI,wa, wL,wS,wA]); wTT = np.array([wT,wT*avd])  # default for comp_N_?

oF_ = ['comp_N_','comp_C_','comp_N','comp_F',  # comp_ functions
       'get_exemplars','cluster_N','cluster_C','cluster_P',  # clust_ functions
       'cross_comp','frame_H','vect_edge','trace_edge','ffeedback','proj_N','comp_slice','slice_edge']  # combined, ancillary
gF_ = []  # add gating oFs, was nF='E'
# func | block cost,gain,distribution, cost = oF_complexity / vt_complexity, ||oF_, *=data:
Fc_ = [13,11,18,5,4,22,20,12,3,13,14,11,3,5,4,3]; cN_,cC_,cN,cF, cE,ccN,ccC,ccP, cX,cFrm,cVct,cTrc,cBac,cPrj,cCS,cSE = Fc_
Fw_ = copy(Fc_); wN_,wC_,wN,wF, wE,wcN,wcC,wcP, wX,wFrm,wVct,wTrc,wBac,wPrj,wCS,wSE = Fw_  # ave gain/call, init = cost
FTT_= [deepcopy(wTT) for _ in range(16)]; ttN_,ttC_,ttN,ttF, ttE,ttcN,ttcC,ttcP, ttX,ttFrm,ttVct,ttTrc,ttBac,ttPrj,ttCs,ttSE = FTT_
# eval V = Ew - Ec * ave:
Ec_ = [3,3,4,2,1,5,5,4,1,4,4,4,2,2,1.1]  # complexity placeholders || oF_, same evals for different oFs?
Ew_ = copy(Ec_)  # ave gain/eval, init = cost, then evaled_block_w - default_block_w
ETT_= [deepcopy(wTT) for _ in range(16)]  # for more precise eeval?

cN_,cC_,cN,cF, cE,ccN,ccC,ccP, cX,cFrm,cVct,cTrc,cBac,cPrj,cCS,cSE = Fc_
wN_,wC_,wN,wF, wE,wcN,wcC,wcP, wX,wFrm,wVct,wTrc,wBac,wPrj,wCS,wSE = Fw_  # ave gain/call, init = cost
ttN_,ttC_,ttN,ttF, ttE,ttcN,ttcC,ttcP, ttX,ttFrm,ttVct,ttTrc,ttBac,ttPrj,ttCs,ttSE = FTT_  # add ETT_?