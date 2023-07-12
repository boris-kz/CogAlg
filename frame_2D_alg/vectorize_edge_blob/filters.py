ave_inv = 20  # ave inverse m, change to Ave from the root intra_blob?
ave = 5  # ave direct m, change to Ave_min from the root intra_blob?
ave_g = 30  # change to Ave from the root intra_blob?
ave_ga = 0.78  # ga at 22.5 degree
flip_ave = .1
flip_ave_FPP = 0  # flip large FPPs only (change to 0 for debug purpose)
div_ave = 200
ave_rmP = .7  # the rate of mP decay per relative dX (x shift) = 1: initial form of distance
ave_ortho = 20
aveB = 50  # same for m and d?
# comp_param coefs:
ave_dI = ave_inv
ave_M = ave  # replace the rest with coefs:
ave_Ma = 10
ave_G = 10
ave_Ga = 2  # related to dx?
ave_L = 10
ave_x = 1
ave_dx = 5  # inv, difference between median x coords of consecutive Ps
ave_dy = 5
ave_daxis = 2
ave_dangle = 2  # vertical difference between angles
ave_daangle = 2

ave_mval = ave_dval = 10  # should be different
vaves = [ave_mval, ave_dval]
ave_Pm = ave_Pd = 10
P_aves = [ave_Pm,ave_Pd]
ave_PPm = ave_PPm = 10
PP_aves = [ave_PPm,ave_PPm]
ave_Gm = ave_Gd = 10
G_aves = [ave_Gm,ave_Gd]

ave_splice = 10
ave_nsubt = [3,3]
ave_sub = 2  # cost of calling sub_recursion and looping
ave_agg = 3  # cost of agg_recursion
ave_overlap = 10
ave_distance = 3
med_decay = .5  # decay of induction per med layer
PP_vars = ["I", "M", "Ma", "angle", "aangle", "G", "Ga", "x", "L"]
aves = [ave_dI, ave_M, ave_Ma, ave_dangle, ave_daangle, ave_G, ave_Ga, ave_dx, ave_L, ave_mval, ave_dval]
ave_rotate = 10