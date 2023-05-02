def sub_recursion_eval(root):  # for PP or dir_blob

    root_PPm_, root_PPd_ = root.rlayers[-1], root.dlayers[-1]
    for fd, PP_ in enumerate([root_PPm_, root_PPd_]):
        mcomb_layers, dcomb_layers, PPm_, PPd_ = [], [], [], []

        for PP in PP_:  # fd = _P.valt[1]+P.valt[1] > _P.valt[0]+_P.valt[0]  # if exclusive comp fork per latuple|vertuple?
            if fd: comb_layers = dcomb_layers; PP_layers = PP.dlayers; PPd_ += [PP]
            else:  comb_layers = mcomb_layers; PP_layers = PP.rlayers; PPm_ += [PP]
            # fork val, rdn:
            val = PP.valt[fd]; alt_val = sum([alt_PP.valt[fd] for alt_PP in PP.alt_PP_]) if PP.alt_PP_ else 0
            ave = PP_aves[fd] * (PP.rdnt[fd] + 1 + (alt_val > val))
            if val > ave and len(PP.P__) > ave_nsub:
                sub_recursion(PP)  # comp_P_der | comp_P_rng in PPs -> param_layer, sub_PPs
                ave*=2  # rdn incr, splice deeper layers between PPs into comb_layers:
                for i, (comb_layer, PP_layer) in enumerate(zip_longest(comb_layers, PP_layers, fillvalue=[])):
                    if PP_layer:
                        if i > len(comb_layers) - 1: comb_layers += [PP_layer]  # add new r|d layer
                        else: comb_layers[i] += PP_layer  # splice r|d PP layer into existing layer
            # sub_PPs / sub+?
            # include empty comb_layers: # revise?
            if fd:
                PPmm_ = [PPm_] + mcomb_layers; mVal = sum([PP.valt[0] for PP_ in PPmm_ for PP in PP_])
                PPmd_ = [PPm_] + dcomb_layers; dVal = sum([PP.valt[1] for PP_ in PPmd_ for PP in PP_])
                root.dlayers = [PPmd_,PPmm_]
            else:
                PPdm_ = [PPm_] + mcomb_layers; mVal = sum([PP.valt[0] for PP_ in PPdm_ for PP in PP_])
                PPdd_ = [PPd_] + dcomb_layers; dVal = sum([PP.valt[1] for PP_ in PPdd_ for PP in PP_])
                root.rlayers = [PPdm_, PPdd_]
            # root is PP:
            if isinstance(root, CPP):
                for i in 0,1:
                    root.valt[i] += PP.valt[i]  # vals
                    root.rdnt[i] += PP.rdnt[i]  # ad rdn too?
            else:  # root is Blob
                if fd: root.G += sum([alt_PP.valt[fd] for alt_PP in PP.alt_PP_]) if PP.alt_PP_ else 0
                else:  root.M += PP.valt[fd]


def sub_recursion(PP):  # evaluate PP for rng+ and der+

    P__  = [P_ for P_ in reversed(PP.P__)]  # revert bottom-up to top-down
    P__ = comp_P_der(P__) if PP.fds[-1] else comp_P_rng(P__, PP.rng + 1)   # returns top-down
    PP.rdnt[PP.fds[-1] ] += 1  # two-fork rdn, priority is not known?  rotate?

    cP__ = [copy(P_) for P_ in P__]
    sub_PPm_, sub_PPd_ = form_PP_t(cP__, base_rdn=PP.rdnt[PP.fds[-1]], fds=PP.fds)
    PP.rlayers[:] = [sub_PPm_]
    PP.dlayers[:] = [sub_PPd_]
    sub_recursion_eval(PP)  # add rlayers, dlayers, seg_levels to select sub_PPs
