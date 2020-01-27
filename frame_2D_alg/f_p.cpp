#include <iostream>
#include "f_p.h"
const float AVE = 50.0;

double at(const double *arr, int r, int c) {
    int index = r * 4 + c;
    return arr[index];
}

std::vector<Bunch*> form_P_(const double *dert_, int row, int col)  {
    
    std::vector<Bunch*> P_;

    int I = at(dert_, 0, 0); // 0*4 + 0
    int G = at(dert_, 0, 1); // 0*4 + 1
    int Dy = at(dert_, 0, 2); // 0*4 + 2
    int Dx = at(dert_, 0, 3); // 0*4 + 3

    G -= AVE;
    int x0 = 0;
    int L = 1;

    bool _s = G > 0;

    for(int i=1; i<row; i++) {
        int p = at(dert_, i, 0);
        int g = at(dert_, i, 1);
        int dy = at(dert_, i, 2);
        int dx = at(dert_, i, 3);
        float vg = g - AVE;
        // std::cout<<"vg:"<<vg<<'\n';
        bool s = vg > 0.0;

        if (s != _s) {
            Bunch *P = new Bunch(dert_, I, G, Dy, Dx, x0, L, _s);
            P_.push_back(P);

            // initialize new P:
            x0 = i;
            I = 0.0;
            G = 0.0;
            Dy = 0.0;
            Dx = 0.0;
            L = 0;
        }

        I += p;
        G += vg;
        Dy += dy;
        Dx += dx;
        L += 1;
        _s = s;
        
    }

    Bunch *Q = new Bunch(dert_, I, G, Dy, Dx, x0, L, _s);
    P_.push_back(Q);
    return P_;
}
