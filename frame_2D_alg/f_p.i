%module f_p
#define SWIGPYTHON_BUILTIN

%{
  #include "numpy/arrayobject.h"
  #define SWIG_FILE_WITH_INIT  /* To import_array() below */
  #include "f_p.h"
%}
%include "std_map.i"
%import "std_deque.i" 
%import "std_vector.i" 

//%template (mapiv) std::map<char,float>;
//%template () std::vector<Bunch*>;
%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {
  (const double* dert_, int row, int col)
}


%typemap(out) std::vector<Bunch*> {

  for(int i=0; i < $1.size(); ++i) {  
    PyObject *o = PyDict_New();
    
  int count = 0; 

    PyDict_SetItemString(o, "I", PyFloat_FromDouble($1.data()[i]->I));
    PyDict_SetItemString(o, "G", PyFloat_FromDouble($1.data()[i]->G));
    PyDict_SetItemString(o, "Dy", PyFloat_FromDouble($1.data()[i]->Dy));
    PyDict_SetItemString(o, "Dx", PyFloat_FromDouble($1.data()[i]->Dx));
    PyDict_SetItemString(o, "L", PyLong_FromDouble($1.data()[i]->size));
    PyDict_SetItemString(o, "x0", PyLong_FromDouble($1.data()[i]->x0));
    PyDict_SetItemString(o, "sign", PyBool_FromLong($1.data()[i]->sign));

    
    PyObject *outer = PyList_New($1.data()[i]->size);
    for(int j=0; j < $1.data()[i]->size; ++j) { 

      PyObject *inner = PyList_New(4);

      for(int k=0; k < 4; ++k) {  
        PyList_SetItem(inner, k, PyLong_FromDouble($1.data()[i]->dert_[count]));
        count++;
      }
      PyList_SetItem(outer, j, inner);
    }
    PyDict_SetItemString(o, "dert_", outer);
    

    $result =  SWIG_Python_AppendOutput($result, o);
  }
}
%include "f_p.h"




