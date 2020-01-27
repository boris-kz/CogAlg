rm -f *.o f_p_wrap.cpp _f_p.so f_p.py
rm -rf __pycache__

g++ -std=c++11 -O3 -march=native -fPIC -c f_p.cpp
swig -python -c++ -o f_p_wrap.cpp f_p.i

# Next, compile the wrapper code:

g++ -std=c++11 -O3 -march=native -w -fPIC -c $(pkg-config --cflags --libs python3) -I $( python -c "import numpy ; print(numpy.get_include())" ) f_p.cpp f_p_wrap.cpp

g++ -std=c++11 -O3 -march=native -shared f_p.o f_p_wrap.o -o _f_p.so -lm