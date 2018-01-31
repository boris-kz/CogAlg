# Python logic comparison tests
# To help developers when in doubt.
# For level_1_2D_draft.py
# Author: Todor Arnaudov 31.1.2018
'''
0 typ=0, a shold be 0
1 typ=1, a shold be 1
(0, 0, 0) alt is evaluated as not 0
(1, 0, 0) alt is evaluated as not 0
(0, 1, 0) alt is evaluated as not 0
(0, 0, 1) alt is evaluated as not 0
(0, 0, 1) alt is evaluated as not 0
[0, 0, 0] alt is evaluated as False (0)
[1, 0, 0] alt is evaluated as True (not 0)
[0, 1, 0] alt is evaluated as True (not 0)
[0, 0, 1] alt is evaluated as True (not 0)
[-1, 0, 0] alt is evaluated as True (not 0)
[0, -1, 0] alt is evaluated as True (not 0)
[0, 0, -1] alt is evaluated as True (not 0)
'tuple' object does not support item assignment
Successful reassignment vP =  (0, 0, 0, 0, 0, 0, 0, 1, [])
Successful reassignment dP =  (0, 0, 0, 0, 0, 0, 0, 0, [])
Successful reassignment vP =  (0, 0, 0, 0, 0, 0, 0, 1, [])
Successful reassignment dP =  (0, 0, 0, 0, 0, 0, 0, 99, [])
'''

# Shows the comparison logic of non-zero element in a tuple
# if alt: ... shouldn't be used, it's always True (including 0,0,0)
# I assume that in the draft it was meant to mean: if any of alt components is !=0
def f2():
 alt = 0,0,0
 if (alt): print(alt, "alt is evaluated as not 0")
 alt = 1,0,0
 if (alt): print(alt, "alt is evaluated as not 0")
 alt = 0,1,0
 if (alt): print(alt, "alt is evaluated as not 0")
 alt = 0,0,1
 if (alt): print(alt, "alt is evaluated as not 0")

 if (alt): print(alt, "alt is evaluated as not 0")
 alt = 1,0,0

 alts = ([0,0,0], [1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1])
 for alt in alts: 
   if (alt[0] or alt[1] or alt[2]): print(alt, "alt is evaluated as True (not 0)")
   else: print(alt, "alt is evaluated as False (0)")

def f3():
 #ycomp
 vP = 0,0,0,0,0,0,0,0,[]  # pri_s, I, D, Dy, M, My, G, alt_rdn, e_
 dP = 0,0,0,0,0,0,0,0,[]  # pri_s, I, D, Dy, M, My, G, alt_rdn, e_

 alt = 1, 45, 24 
 alt_len, alt_vG, alt_dG = alt  # same for vP and dP, incomplete vg - ave / (rng / X-x)?
 #alt_vG > alt_dG --> vP[7] = 1 (0,0,0,0,0,0,0,1,[])
 try:
   if alt_vG > alt_dG:  # comp of alt_vG to alt_dG, == goes to alt_P or to vP: primary?
                    vP[7] += alt_len  # alt_len is added to redundant overlap of lesser-oG-  vP or dP
   else: dP[7] += alt_len  # no P[8] /= P[7]: rolp = alt_rdn / len(e_): P to alt_Ps rdn rati
 except TypeError as err: print(err)
 
  
 try:
   if alt_vG > alt_dG:  # comp of alt_vG to alt_dG, == goes to alt_P or to vP: primary?
                    vP = vP[0], vP[1], vP[2], vP[3], vP[4], vP[5], vP[6], vP[7] + alt_len, vP[8]  # alt_len is added to redundant overlap   of lesser-oG-  vP or dP
   else: dP = dP[0], dP[1], dP[2], dP[3], dP[4], dP[5], dP[6], dP[7] + alt_len, dP[8]  # no P[8] /= P[7]: rolp = alt_rdn / len(e_): P to alt_Ps rdn rati
   print("Successful reassignment vP = ", vP)
   print("Successful reassignment dP = ", dP)
 except TypeError as err: print(err)

 alt = 99, 25, 56 #now alt_dG >= alt_vG, update dP to (0,0,0,0,0,0,0,99,[])
 alt_len, alt_vG, alt_dG = alt 
 try:
   if alt_vG > alt_dG:  # comp of alt_vG to alt_dG, == goes to alt_P or to vP: primary?
                    vP = vP[0], vP[1], vP[2], vP[3], vP[4], vP[5], vP[6], vP[7] + alt_len, vP[8]  # alt_len is added to redundant overlap   of lesser-oG-  vP or dP
   else: dP = dP[0], dP[1], dP[2], dP[3], dP[4], dP[5], dP[6], dP[7] + alt_len, dP[8]  # no P[8] /= P[7]: rolp = alt_rdn / len(e_): P to alt_Ps rdn rati
   print("Successful reassignment vP = ", vP)
   print("Successful reassignment dP = ", dP)
 except TypeError as err: print(err)


#if typ: when scalar numerical is correctly evaluated (as in C), 0 = False, not 0 = True
def f1():
 typ = 0
 a = 5

 if typ: a=1
 else: a=0
 print(a, "typ=0, a shold be 0")

 typ = 1
 if typ: a=1
 else: a=0
 print(a, "typ=1, a shold be 1")

#####

f1()
f2()
f3()
