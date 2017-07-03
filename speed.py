# speed.py
# Performance tests for lists, deques, iterators etc.
# for tuning the CogAlg implementations.
# Date/Version  : 3-7-2017
# Author        : Todor Iliev Arnaudov - Tosh/Twenkid,  http://artificial-mind.blogspot.bg
#
#Sample run:
#time twoLists 0.025516033172607422
#time twoLists_one_pop 0.05803799629211426
#time traverse_list 0.005003929138183594
#time pop_list 0.028517961502075195
#time append_list 0.03252267837524414
#time append_deque 0.027525663375854492
#time appendleft_deque 0.030516862869262695
#time deque_pop_left 0.03302145004272461

from scipy import misc
import numpy as np
#import timeit
import time
from collections import deque

def list(data, count):
 t_ = []
 for i in range(0, len(data),3):
   t = data[i], data[i+1], data[i+2]
  # print(i); print(t)
   t_.append(t)
 #print(t_)
 return t_ 
 
 
def twoLists(t_, _t_):

 for t, _t in zip(t_, _t_):
  #print(t, _t)
   p, m, d = t
   _p, _m, _d = _t
 #return time
 
def twoLists_one_pop(t_, _t_):

 for t in t_:
  #print(t, _t)
   p, m, d = t
   _t = _t_.pop()
   _p, _m, _d = _t
   
 
#call: list, list --> 
def appendleft_deque(t_):
 d = deque()
 for i in t_:
   d.appendleft(i)
 return d
   
#call: list, list --> 
def append_deque(t_):
 d = deque()
 for i in t_:
   d.append(i)
 return d
 
 #call: list, list --> 
def append_list(t_):
 #print(len(t_))
 l = []
 for i in t_:
   l.append(i)
 return l
 
def pop_list(t_):
 for i in range(0, len(t_)):
   s = t_.pop()
 
def traverse_list(t_):
 for i in t_:
   s = i

#call with a deque
def deque_pop_left(t_):
   for i in range(0,len(t_)):
     s = t_.popleft()
   
def zip1():   
#t_ = [(1,5,11),(2,10, 15), (3,4,5)]
#_t_ = [(124,15, 5), (199, 20, 4), (240, 45, 31)]
 for t, _t in zip(t_, _t_):
  print(t, _t)
  p, d, m = t
  _p, _d, _m = _t
  print(p,d,m)
  print(_p,_d,_m)
  
 
  #for t, _t in zip(t_, _t_):
  
  
def main():
  n = 1000 
  f = misc.face(gray=True)  # input frame of pixels
  f = f.astype(int) 
  x, y = f.shape
 # print(f.shape)
  f2 = np.reshape(f, (x*y)) # x*y)
  #print(f2.shape)
  #print("face")
 # print(f2)
  tA = list(data = f2, count = n)
  tB = list(data = f2, count = n)
  
  tC = tA.copy()
  
  #print("tA", tA)
  #print("tB", tB)
  #print(timeit.timeit("twoLists(t_ = tupleA, _t_ = tupleB)", setup="from __main__ import twoLists")) 
  start = time.time()
  twoLists(t_ = tA, _t_ = tB)
  period = time.time() - start
  print("time twoLists", period)
  
  start = time.time()
  twoLists_one_pop(t_ = tA, _t_ = tB)
  period = time.time() - start
  print("time twoLists_one_pop", period)
  #tB is now empty, consumed by pop_list
  
  start = time.time()
  traverse_list(t_ = tC)
  period = time.time() - start
  print("time traverse_list", period)
  
  
  start = time.time()
  pop_list(t_ = tA)
  period = time.time() - start
  print("time pop_list", period)
  #tA is now empty, consumed by pop_list
  #using the copy tC
  
  start = time.time()
  append_list(t_ = tC)
  period = time.time() - start
  print("time append_list", period)
 
  start = time.time()
  append_deque(t_ = tC)
  period = time.time() - start
  print("time append_deque", period)
  
  start = time.time()
  appendleft_deque(t_ = tC)
  period = time.time() - start
  print("time appendleft_deque", period)
 
  tD = deque(tC)
  start = time.time()
  deque_pop_left(t_ = tD)
  period = time.time() - start
  print("time deque_pop_left", period)
 
if __name__ == '__main__':
  main()
