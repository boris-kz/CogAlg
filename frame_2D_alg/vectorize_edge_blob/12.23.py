for part, epart in zip((valt, rdnt, dect), (evalt, erdnt, edect)):
    for i in 0, 1:  # internalize externals
        part[i] += epart[i]

Val, Rdn = 0, 0
for subH in graph.aggH[1:]:  # eval by sum of val,rdn of top subLays in lower aggLevs:
    if not subH[0]: continue  # empty rim if no prior comp
    Val += subH[0][-1][1][fd]  # subH: [derH, valt,rdnt,dect]
    Rdn += subH[0][-1][2][fd]
