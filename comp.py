# A peek in parameters' count and number of return items overhead
# comp_all:0.6099114418029785(7.769572507044312e-07)[785000 calls]
# comp_ret2:0.5483739376068115(6.985655256137726e-07)[785000 calls]
# comp_4_in:0.24616742134094238(3.1358907177190113e-07)[785000 calls]
# comp_all_nop:0.537372350692749(6.845507652136931e-07)[785000 calls]
# comp_ret2_nop:0.4793403148651123(6.1062460492371e-07)[785000 calls]
# comp_4_in_nop:0.17783737182617188(2.2654442270849922e-07)[785000 calls]
# In another test: 
# misc.face-astypeint:0.19313502311706543(0.19313502311706543)[1 calls]
# Le1:5.379834890365601(5.379834890365601)[1 calls]
# On that rough and synthetic estimate, calling overhead of comp may be about 10% of the processing time for Le1
# Date: 5-7-2017
# Author: Todor
# https://github.com/Twenkid/CogAlg/


import time

def comp_all(p, pri_p, fd, fv, W, x, 
         pri_s, I, D, V, rv, p_,
         pri_sd, Id, Dd, Vd, rd, d_,
         a, aV, aD, min_r, A, AV, AD, r, P_):
	fd+=fv
	vP_ = []; dP_ = [] 
		 
	return pri_s, I, D, V, rv, p_, pri_sd, Id, Dd, Vd, rd, d_, P_


def comp_4_in(p, pri_p, fd, fv):
   #pri_s=0; I=9; D=2466; V=35578; rv=1; p_=; pri_sd; Id; Dd; Vd; rd; d_; P_
	fd+=fv
	vP_ = []; dP_ = [] 
	return vP_, dP_

def comp_ret2(p, pri_p, fd, fv, W, x, 
         pri_s, I, D, V, rv, p_,
         pri_sd, Id, Dd, Vd, rd, d_,
         a, aV, aD, min_r, A, AV, AD, r, P_):
		fd+=fv
		vP_ = []; dP_ = [] 
		return vP_, dP_
		
def showtime(name, end, start, cnt):
	print(name + ":" + str(end-start) + "(" +  str((end-start)/cnt) + ")" + "[" + str(cnt) + " calls]")
	#per = (end-start)/cnt	
	#print(per)	 
	

def comp_all_nop(p, pri_p, fd, fv, W, x, 
         pri_s, I, D, V, rv, p_,
         pri_sd, Id, Dd, Vd, rd, d_,
         a, aV, aD, min_r, A, AV, AD, r, P_):
		 
	return pri_s, I, D, V, rv, p_, pri_sd, Id, Dd, Vd, rd, d_, P_


def comp_4_in_nop(p, pri_p, fd, fv):
   #pri_s=0; I=9; D=2466; V=35578; rv=1; p_=; pri_sd; Id; Dd; Vd; rd; d_; P_

	return fd, fv

def comp_ret2_nop(p, pri_p, fd, fv, W, x, 
         pri_s, I, D, V, rv, p_,
         pri_sd, Id, Dd, Vd, rd, d_,
         a, aV, aD, min_r, A, AV, AD, r, P_):
		
		return fd, fv


p = 24
pri_p = 100
fd = 35; fv = 25;
W = 800;
x = 100;
pri_s = 1; I = 1567; D = 257; V = 3953; rv = 1; p_ = [1, 5, 10, 23, 56, 99]
pri_sd = 1; Id = 4956; Dd = 5423; Vd = 1356; rd = 1; d_ = [0, -5, 9, 12, -7, 9]
a = 127; aV = 4567; aD = 145; min_r = 1; A = 66346; AV = 94545; AD = 355; r = 1; P_ = []

cnt = 785000

#A: Including 1+, 2 creations of empty lists, sometimes returning lists
 
start = time.time()
for i in range(cnt):
  comp_all(p, pri_p, fd, fv, W, x,
         pri_s, I, D, V, rv, p_,
         pri_sd, Id, Dd, Vd, rd, d_,
         a, aV, aD, min_r, A, AV, AD, r, P_)
end = time.time()
showtime("comp_all", end, start, cnt)


start = time.time()
for i in range(cnt):
  comp_ret2(p, pri_p, fd, fv, W, x,
         pri_s, I, D, V, rv, p_,
         pri_sd, Id, Dd, Vd, rd, d_,
         a, aV, aD, min_r, A, AV, AD, r, P_)
		 
end = time.time()
showtime("comp_ret2", end, start, cnt)

start = time.time()
for i in range(cnt):
  comp_4_in(p, pri_p, fd, fv)
end = time.time()
showtime("comp_4_in", end, start, cnt)


#B: empty function bodies

start = time.time()
for i in range(cnt):
  comp_all_nop(p, pri_p, fd, fv, W, x,
         pri_s, I, D, V, rv, p_,
         pri_sd, Id, Dd, Vd, rd, d_,
         a, aV, aD, min_r, A, AV, AD, r, P_)
end = time.time()
showtime("comp_all_nop", end, start, cnt)


start = time.time()
for i in range(cnt):
  comp_ret2_nop(p, pri_p, fd, fv, W, x,
         pri_s, I, D, V, rv, p_,
         pri_sd, Id, Dd, Vd, rd, d_,
         a, aV, aD, min_r, A, AV, AD, r, P_)
		 
end = time.time()
showtime("comp_ret2_nop", end, start, cnt)

start = time.time()
for i in range(cnt):
  comp_4_in_nop(p, pri_p, fd, fv)
end = time.time()
showtime("comp_4_in_nop", end, start, cnt)

