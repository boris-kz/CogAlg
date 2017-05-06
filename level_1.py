from scipy import misc


def inc_rng(a, aV, aD, min_r, A, AV, AD, r, p_):

   if r > min_r:  # A, AV, AD inc.to adjust for redundancy to patterns formed by prior comp:
       A += a     # a: min m for inclusion into positive vP
       AV += aV   # aV: min V for initial comp() recursion, AV: min V for higher recursions

   if r > min_r-1:  # default range is shorter for d_[w]: redundant ds are smaller than ps
       AD += aD     # aV: min |D| for comp() recursion over d_[w], AD: min |D| for recursion

   W = len(p_)
   ip_ = p_  # to differentiate from new p_; local nP is initialized:

   P_ = []  # r was incremented in higher-scope p_
   pri_s, I, D, V, rv, p_ = (0, 0, 0, 0, 0, [])  # tuple vP=0
   pri_sd, Id, Dd, Vd, rd, d_ = (0, 0, 0, 0, 0, [])  # tuple dP=0

   for x in range(r+1, W):

       p, fd, fv = ip_[x]       # compared to a pixel at x-r-1:
       pp, pfd, pfv = ip_[x-r]  # previously compared p(ignored), its fd, fv to next p
       fv += pfv  # fv is summed over extended-comp range
       fd += pfd  # fd is summed over extended-comp range

       pri_p, pri_fd, pri_fv = ip_[x-r-1]  # for comp(p, pri_p), pri_fd and pri_fv ignored

       pri_s, I, D, V, rv, p_, pri_sd, Id, Dd, Vd, rd, d_, P_ = \
       comp(p, pri_p, fd, fv, W, x,
            pri_s, I, D, V, rv, p_,
            pri_sd, Id, Dd, Vd, rd, d_,
            a, aV, aD, min_r, A, AV, AD, r, P_)

   return P_  # local vPs and dPs to replace p_, A, AV, AD accumulated per comp recursion


def inc_der(a, aV, aD, min_r, A, AV, AD, r, d_):

   if r > min_r:
       A += a; AV += aV
   if r > min_r-1:
       AD += aD

   W = len(d_)
   ip_ = d_  # to differentiate from new d_; local nP is initialized:

   fd, fv, r, P_ = (0, 0, 0, [])  # r is initialized for each d_
   pri_s, I, D, V, rv, p_ = (0, 0, 0, 0, 0, [])  # tuple nP=0,
   pri_sd, Id, Dd, Vd, rd, d_ = (0, 0, 0, 0, 0, [])  # tuple nP=0

   for x in range(W):

       p = ip_[x]  # better than pop()?
       if x > 0:

           pri_s, I, D, V, rv, p_, pri_sd, Id, Dd, Vd, rd, d_, P_ = \
           comp(p, pri_p, fd, fv, W, x,
                pri_s, I, D, V, rv, p_,
                pri_sd, Id, Dd, Vd, rd, d_,
                a, aV, aD, min_r, A, AV, AD, r, P_)

       pri_p = p

   return P_  # local vPs and dPs to replace d_


def comp(p, pri_p, fd, fv, W, x,  # x is from higher-scope for loop w
        pri_s, I, D, V, rv, p_,
        pri_sd, Id, Dd, Vd, rd, d_,
        a, aV, aD, min_r, A, AV, AD, r, P_):

   d = p - pri_p      # difference between consecutive pixels
   m = min(p, pri_p)  # match between consecutive pixels
   v = m - A          # relative match (predictive value) between consecutive pixels

   fd += d  # fd includes all shorter + current- range ds between comparands
   fv += v  # fv includes all shorter + current- range vs between comparands

   # formation of value pattern vP: span of pixels forming same-sign v s:

   s = 1 if v > 0 else 0  # s: positive sign of v
   if x > r+2 and (s != pri_s or x == W-1): # if derived pri_s miss, vP is terminated

       if len(p_) > r+3 and pri_s == 1 and V > AV:  # min 3 comp over extended distance within p_:

           r += 1  # r: incremental range of comp
           rv = 1  # rv: incremental range flag:
           p_.append(inc_rng(a, aV, aD, min_r, A, AV, AD, r, p_))

       vP = 0, pri_s, I, D, V, rv, p_
       P_.append(vP)  # output of vP
       I, D, V, rv, p_ = (0, 0, 0, 0, [])  # initialization of new vP

   pri_s = s   # vP (span of pixels forming same-sign v) is incremented:
   I += pri_p  # ps summed within vP
   D += fd     # fds summed within vP into fuzzy D
   V += fv     # fvs summed within vP into fuzzy V
   pri = pri_p, fd, fv
   p_.append(pri)  # buffered within w for selective extended comp

   # formation of difference pattern dP: span of pixels forming same-sign d s:

   sd = 1 if d > 0 else 0  # sd: positive sign of d;
   if x > r+2 and (sd != pri_sd or x == W-1):  # if derived pri_sd miss, dP is terminated

       if len(d_) > 3 and abs(Dd) > AD:  # min 3 comp within d_:

           rd = 1  # rd: incremental derivation flag:
           d_.append(inc_der(a, aV, aD, min_r, A, AV, AD, r, d_))

       dP = 1, pri_sd, Id, Dd, Vd, rd, d_
       P_.append(dP)  # output of dP
       Id, Dd, Vd, rd, d_ = (0, 0, 0, 0, [])  # initialization of new dP

   pri_sd = sd  # dP (span of pixels forming same-sign d) is incremented:
   Id += pri_p  # ps summed within wd
   Dd += fd     # fds summed within wd
   Vd += fv     # fvs summed within wd
   d_.append(fd)  # prior fds are buffered in d_, same-sign as distinct from in p_

   return pri_s, I, D, V, rv, p_, pri_sd, Id, Dd, Vd, rd, d_, P_
   # for next p comparison, vP and dP increment, and output

def L1(Fp_):  # last '_' distinguishes array name from element name

   FP_ = []  # output frame of vPs: relative match patterns, and dPs: difference patterns
   H, W = Fp_.shape  # H: frame height, W: frame width

   a = 127  # minimal filter for vP inclusion
   aV = 63  # minimal filter for incremental-range comp
   aD = 63  # minimal filter for incremental-derivation comp
   min_r=0  # default range of fuzzy comparison, initially 0

   for y in range(H):

       ip_ = Fp_[y, :]  # y is index of new line ip_
       # or p_ = dict(zip(range(len(p_)), p_.values())))

       if min_r == 0: A = a; AV = aV  # actual filters, incremented per comp recursion
       else: A = 0; AV = 0  # if r > min_r

       if min_r <= 1: AD = aD
       else: AD = 0

       fd, fv, r, x, P_ = (0, 0, 0, 0, [])  # nP tuple = (P_, r, A, AV, AD)
       pri_s, I, D, V, rv, p_ = (0, 0, 0, 0, 0, [])  # vP tuple
       pri_sd, Id, Dd, Vd, rd, d_ = (0, 0, 0, 0, 0, [])  # dP tuple

       for x in range(W):  # cross-compares consecutive pixels, outputs sequence of d, m, v:

           p = ip_[x]  # could use pop()? new pixel, comp to prior pixel:
           if x > 0:

               pri_s, I, D, V, rv, p_, pri_sd, Id, Dd, Vd, rd, d_, P_ = \
               comp(p, pri_p, fd, fv, W, x,
                    pri_s, I, D, V, rv, p_,
                    pri_sd, Id, Dd, Vd, rd, d_,
                    a, aV, aD, min_r, A, AV, AD, r, P_)

           pri_p = p  # prior pixel, pri_ values are always derived before use

       FP_.append(P_)  # line of patterns P_ is added to frame of patterns, y = len(FP_)

   return FP_  # or return FP_  # frame of patterns (vP or dP): output to level 2;

f = misc.face(gray=True)  # input frame of pixels
f = f.astype(int)
L1(f)

# print ('type', 0, 'pri_s', pri_s, 'I', I, 'D', D, 'V', V, 'rv', rv, 'p_', p_)
# print ('type', 1, 'pri_sd', pri_sd, 'Id', Id, 'Dd', Dd, 'Vd', Vd, 'rd', rd, 'd_', d_)

