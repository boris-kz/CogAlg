def weigh_m_(m_, M, ave):  # adjust weights on attr matches, also add cost attrs

    L = len(m_)
    M = np.sqrt(sum([m ** 2 for m in m_]) / L)
    _w_ = [1 for _ in m_]
    while True:
        w_ = [m/M for m in m_]  # rational deviations from mean
        Dw = sum([abs(w - _w) for w, _w in zip(w_, _w_)])  # weight update
        M = sum((m if m else 1e-7) * w for m, w in zip(m_, w_)) / L  # M update
        if Dw > ave:
            _w_ = w_
        else:
            break
    return w_, M

L = comp_N(hG, lev_G, rn=_n / n if _n > n else n / _n)
if Val_(L.Et, _Et=L.Et) > 0:

# m/mag per attr:
rm_ = np.divide(hG.vert[0], hG.latuple)  # summed from all layers: no need for base vert?
rm_ = np.divide(hG.vert[0], hG.vert[0] + np.abs(hG.vert[1]))
for lay in hG.derH:
    for fork in lay:  # add rel CLay.m_d_t[0], or vert m_=rm_?
        rm_ += np.divide(fork.m_d_t[0], fork.m_d_t[0] + np.abs(fork.m_d_t[1]))
