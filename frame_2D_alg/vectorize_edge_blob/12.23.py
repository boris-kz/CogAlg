for part, epart in zip((valt, rdnt, dect), (evalt, erdnt, edect)):
    for i in 0, 1:  # internalize externals
        part[i] += epart[i]

Val, Rdn = 0, 0
for subH in graph.aggH[1:]:  # eval by sum of val,rdn of top subLays in lower aggLevs:
    if not subH[0]: continue  # empty rim if no prior comp
    Val += subH[0][-1][1][fd]  # subH: [derH, valt,rdnt,dect]
    Rdn += subH[0][-1][2][fd]

def sum_Hts(ValHt,RdnHt,DecHt, valHt,rdnHt,decHt):
    # loop m,d Hs, add combined decayed lower H/layer?
    for ValH,valH, RdnH,rdnH, DecH,decH in zip(ValHt,valHt, RdnHt,rdnHt, DecHt,decHt):
        ValH[:] = [V+v for V,v in zip_longest(ValH, valH, fillvalue=0)]
        RdnH[:] = [R+r for R,r in zip_longest(RdnH, rdnH, fillvalue=0)]
        DecH[:] = [D+d for D,d in zip_longest(DecH, decH, fillvalue=0)]
'''
derH: [[tuplet, valt, rdnt, dect]]: default input from PP, rng+|der+, sum min len?
subH: [derH_t]: m,d derH, m,d ext added by agg+ as 1st tuplet
aggH: [subH_t]: composition levels, ext per G, 
'''
