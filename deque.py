#Debug expression:  init = 0 if select_ == 1 else 1
#I think it should be: len(select_) or  len(select_)>0  (for more than one elements)
#30-1-2018

from collections import deque

select_ = deque()

print("==========================")
init = 0
init = 0 if select_ == 1 else 1
print( init, "Empty deque, expected 0", select_);
print("==========================")

select_ .append(5)

init = 0 if select_ == 1 else 1
print(init, "Deque with elements, expected 1");
print(select_)
print("==========================")
select_ = None

init = 0 if select_ == 1 else 1
print(init, "None variable");
print(select_)

print("==========================")
select_= deque()
init = 0 if len(select_) == 1 else 1
print(init, "len(select_) on empty list, expected 1");
print(select_)

print("==========================")
select_.append([5,66,12])
init = 0 if len(select_) == 1 else 1
print(init, "len(select_) on list with one element, expected 0");
print(select_)


