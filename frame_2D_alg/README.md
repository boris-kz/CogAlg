# Building the Native Libraries

- Ensure you have a valid c++11 library
- Ensure [swig](http://www.swig.org/download.html) is installed on your system
- Numpy core libraries should be installed
- Open terminal and give the following command

```sh
$ bash build.sh
```

This shall generate many files. 2 most important files are `_f_p.so` and `f_p.py`.

- Make sure you have these two files in your local directory.
- Now run the `frame_blob.py` file. We have modified the code to accept the native implementation of frame_blobs.


# More things to do

- Improve then memory management of the C++ code. (Now we are not deallocating the memory)
- Implement the `scan_p` in C++ and integrate it too.
