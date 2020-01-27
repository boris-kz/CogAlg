/* File knn.h */
#ifndef F_P_H
#define f_P_H
#define CPP_14 0

#include <memory> 
#include <vector>
#include <memory> 
#include <algorithm>
#include <tuple>
#include <initializer_list>
#include <iostream>

class Bunch {
    public:

        float I;
        float G;
        float Dy;
        float Dx;
        int x0;
        bool sign;
        size_t size;        
        std::shared_ptr<float[]> dert_;
        explicit Bunch(const double *Dert_, float I, float G, float Dy, float Dx, int x0, size_t sz, bool _s) {
            this->I = I;
            this->G = G;
            this->Dy = Dy;
            this->Dx = Dx;
            this->size = sz;
            this->x0 = x0;
            this->sign = _s; 
            #if CPP_14
                this->dert_ = std::make_shared<float[]>(this->size*4);
            #else
                this->dert_ = std::shared_ptr<float[]>(new float[this->size*4]);
            #endif

            for(int i=0; i< this->size*4; i++) {
                this->dert_[i] = Dert_[x0*4+i];
            }
        }

};
/* Define function prototype */
std::vector<Bunch*> form_P_(const double *dert_, int row, int col)  ;
#endif