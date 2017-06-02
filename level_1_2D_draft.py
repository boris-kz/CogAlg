from scipy import misc

'''

Level 1 with 2D gradient: modified combination of core algorithm levels 1 and 2 

Initial 2D comparison forms lateral and vertical derivatives: 2 matches and 2 differences per pixel. Both lateral and vertical comparison is

performed on the same level because average lateral match ~ average vertical match. These derivatives form quadrant gradients:

average of rightward and downward match or difference per pixel (they equally representative samples of quadrant).

Quadrant gradient is a minimal unit of 2D gradient, so 2D pattern is defined by matching sign of 

quadrant gradient of value for vP, or quadrant gradient of difference for dP.

'''

def comp(p, pri_p, pri_py, fd, fv, x, y, W, H, A, r, vP_, dP_,  # x and y are from higher-scope for loop w, h
         pri_s, I, D, V, rv, p_, ow, alt_,
         pri_sd, Id, Dd, Vd, rd, d_, owd, dalt_,
         pri_sy, # summation is per 1D P, them comp? or all sums are 2D?
         pri_sdy
         ):

    d = p - pri_p    # difference between consecutive pixels
    dy = p - pri_py  # difference between vertically consecutive pixels
    dq = d + dy      # quadrant gradient of difference
    fd += dq         # all shorter + current- range dq s within extended quadrant?

    m = min(p, pri_p)  # match between consecutive pixels
    my = min(p, pri_py)  # match between vertically consecutive pixels
    vq = m + my - A    # quadrant gradient of predictive value (relative match)
    fv += vq           # all shorter + current- range vq s within extended quadrant

    # formation of 1D value pattern vP: span of pixels forming same-sign v s:

    s = 1 if vq > 0 else 0  # s: positive sign of v
    if x > r+2 and (s != pri_s or x == W-1):  # if derived pri_s miss, vP is terminated

        vP = pri_s, I, D, V, rv, p_, alt_  # no default div: redundancy eval per P2 on Le2?

        # P term, formation of 2D value pattern vP2: blob of pixels forming same-sign v s:
        # comp to vertically prior Ps in _P_:
        # synchronized with input p? or pri_py, to be terminated?
        # addition to P2

        vP_.append(vP)  # vP_ is within vP2,

        alt = len(vP_), ow  # len(P_) is an index of last overlapping vP
        alt_.append(alt)  # addresses of overlapping vPs and ow are buffered at current dP

        I, D, V, rv, ow, owd, p_, alt_ = (0, 0, 0, 0, 0, 0, [], [])  # initialization of new vP and ow

    pri_s = s   # vP (span of pixels forming same-sign v) is incremented:
    ow += 1     # overlap to current dP
    I += pri_p  # ps summed within vP
    D += dq     # fds summed within vP into fuzzy D
    V += fv     # fvs summed within vP into fuzzy V
    pri = pri_p, dq, fv
    p_.append(pri)  # buffered within w for selective extended comp


    if y > r+2 and (s != pri_sy or y == H-1):  # if derived pri_sy miss, vP2 is terminated,

       # also eval of rotate, 1D re-scan and re-comp

    # formation of difference pattern dP: span of pixels forming same-sign d s:

    sd = 1 if d > 0 else 0  # sd: positive sign of d;
    if x > r+2 and (sd != pri_sd or x == W-1):  # if derived pri_sd miss, dP is terminated

        pd = Id / len(d_); dd = Dd / len(d_); vd = Vd / len(d_)  # all alt Ps can be immediately evaluated?
        dP = 1, pri_sd, pd, Id, dd, Dd, vd, Vd, rd, d_, dalt_
        dP_.append(dP)  # output of dP

        dalt = len(dP_), owd  # len(P_) is an address of last overlapping dP
        dalt_.append(dalt)  # addresses of overlapping dPs and owds are buffered at current vP

        Id, Dd, Vd, rd, ow, owd, d_, dalt_ = (0, 0, 0, 0, 0, 0, [], [])  # initialization of new dP and ow

    pri_sd = sd  # dP (span of pixels forming same-sign d) is incremented:
    owd += 1     # overlap to current vP
    Id += pri_p  # ps summed within wd
    Dd += fd     # fds summed within wd
    Vd += fv     # fvs summed within wd
    d_.append(fd)  # prior fds are buffered in d_, same-sign as distinct from in p_

    return pri_s, I, D, V, rv, p_, ow, alt_, pri_sd, Id, Dd, Vd, rd, d_, owd, dalt_, vP_, dP_
    # for next p comparison, vP and dP increment, and output

'''

1D orientation is initially arbitrary but comparison of average coordinate x and width w is necessary to re-orient it for maximized 1D P match.
 
- initial 1D Ps are not nested, and are not defined by horizontal sign,
 
- elements of p_ (but not d_) contain both leftward and downward derivatives,
 
- overlap () records area of overlap per alt_P2s, assigned to both and selected at 2Le eval 
 
 
2D P includes array of initial 1D Ps and their variables, external vars x and w (new first) are cross-compared within 2D P (P2) by default:
 
 
comp (x, _x) -> dx; mx = mean_dx - dx // normalized compression of distance: min cost decrease, not min benefit?
 
  if dx > a: comp (|dx|) // or if dxP Dx: fixed ddx cost?  comp of same-sign dx only
 
comp (w, _w) -> mw, dw // default, re-orientation if projected difference decr. / match incr. for minimized 1D Ps over maximized 2D:
 
  comp (dw, ddx);  if match (dw, ddx) > a: _w *= cos (ddx); comp (w, _w) // proj w * cos match, if same-sign dw, ddx, not |.|
 
 
if mx+mw > a: // conditional input vars norm and comp, also at P2 term: rotation if match (-DS, Ddx), div_comp if rw?  
 
   comp (dw, ddx) -> m_dw_ddx // to angle-normalize S vars for comp:
 
   if m_dw_ddx > a: _S /= cos (ddx)
 
   if dw > a: div_comp (w) -> rw // to width-normalize S vars for comp: 
 
      if rw > a: pn = I/w; dn = D/w; vn = V/w; 
 
         comp (_n) // or default norm for redun assign, but comp (S) if low rw?
 
         if d_n > a: div_comp (_n) -> r_n // or if d_n * rw > a: combined div_comp eval: ext, int co-variance?
 
   else: comp (S) // even if norm for redun assign?
 
 
P2 term: orient and norm at a fixed cost of *cos and div_comp: two-level selection?
 
 
L = h (height) / cos (Dx / h);  w = W/ L // max L, min ave w, within level 1?  
 
if yD * M_dw_ddx > a: scan angle * cos (Dx) // or if match (-DS, Ddx)? 
 
   coord stretch | compress \ ddx, or rotation: shift and re-scan \ Ddx, or re-definition \ frame Ddx,  to min difference and max match?
 
   potentially recursive 1D ) 2D comp within P2: generic levels 1 and 2 
 
 
if Mx + Mw > a: // /+ extv_P2 of any power, value of summed-variables S (xI, xD, xM) norm and comp between 1D Ps:
 
   if SS * Dw > a: div_comp (w, S) // initial P comp override?
 
   else: comp (S, _S)

'''

def Le1(Fp_): # last '_' distinguishes array name from element name

    FP_ = []  # output frame of vPs: relative match patterns, and dPs: difference patterns
    H, W = Fp_.shape  # H: frame height, W: frame width

    a = 127  # minimal filter for vP inclusion
    min_r=0  # default range of fuzzy comparison, initially 0

    for y in range(H):

        ip_ = Fp_[y, :]  # y is index of new line ip_
        if y > 0: _ip_ = Fp_[y-1, :]  # _ip_: higher line of pixels
        else: _ip_ = 0  # W * 0?  

        if min_r == 0: A = a
        else: A = 0;

        fd, fv, r, x, vP_, dP_ = (0, 0, 0, 0, [], [])  # nP tuple
        pri_s, I, D, V, rv, ow, p_, alt_ = (0, 0, 0, 0, 0, 0, [], [])  # vP tuple
        pri_sd, Id, Dd, Vd, rd, owd, d_, dalt_ = (0, 0, 0, 0, 0, 0, [], [])  # dP tuple

        for x in range(W):  # cross-compares consecutive pixels, outputs sequence of d, m, v:

            p = ip_[x]  # could use pop()? new pixel, comp to prior pixel:
            _p = _ip_[x]  # also contains patterns:

            if x > 0:
                pri_s, I, D, V, rv, p_, ow, alt_, pri_sd, Id, Dd, Vd, rd, d_, owd, dalt_, vP_, dP_ = \
                comp(p, pri_p, fd, fv, W, x, A, r, vP_, dP_,
                pri_s, I, D, V, rv, p_, ow, alt_,
                pri_sd, Id, Dd, Vd, rd, d_, owd, dalt_)

            pri_p = p  # prior pixel, pri_ values are always derived before use

        LP_ = vP_, dP_
        FP_.append(LP_)  # line of patterns P_ is added to frame of patterns, y = len(FP_)

    return FP_  # or return FP_  # frame of patterns (vP or dP): output to level 2;

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
Le1(f)
