'''
            for G in link._G, link.G:
                # draft
                if len_root_HH > -1:  # agg_compress, increase nesting
                    if len(G.rim_t)==len_root_HH:  # empty rim layer, init with link:
                        if fd:
                            G.Vt=[0,Val]; G.Rt=[0,Rdn]; G.Dt=[0,Dec]  # temporary
                            G.rim_t = [[[[[],[link]]]],2]; G.rim_t = [[[[[],[link]]]],2]  # init rim_tHH, depth = 2
                        else:
                            G.Vt=[Val,0]; G.Rt=[Rdn,0]; G.Dt=[Dec,0]
                            G.rim_t = [[[[[link],[]]]],2]; G.rim_t = [[[[[link],[]]]],2]
                    else:
                        G.Vt[fd] += Val; G.Rt[fd] += Rdn; G.Dt[fd] += Dec  # accum rim layer with link
                        G.rim_t[0][-1][-1][fd] += [link]; G.rim_t[0][-1][-1][fd] += [link]  # append rim_tHH
                else:
                    if len(G.rim_t)==len_root_H:  # empty rim layer, init with link:
                        if fd:
                            G.Vt=[0,Val]; G.Rt=[0,Rdn]; G.Dt=[0,Dec]
                            G.rim_t = [[[[],[link]]],1]; G.rim_t = [[[[],[link]]],1]  # init rim_tH, depth = 1
                        else:
                            G.Vt=[Val,0]; G.Rt=[Rdn,0]; G.Dt=[Dec,0]
                            G.rim_t = [[[[link],[]]],1]; G.rim_t = [[[[link],[]]],1]
                    else:
                        G.Vt[fd] += Val; G.Rt[fd] += Rdn; G.Dt[fd] += Dec  # accum rim layer with link
                        G.rim_t[0][-1][fd] += [link]; G.rim_t[0][-1][fd] += [link]  # append rim_tH
'''