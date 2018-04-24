# CogAlg


## Alorithm explanation

0. Import OpenCV 3.0.0 to read image in gray scale format as 2d-array and argument parser to get image path as input parameter.
1. Initialize argument parser with `--image` input parameter, where parameter represent path to input image from initial folder, where script is running.
2. Initialize pattern filters here as constants for high-level feedback from algorithm.
3. Run root `image_to_patterns` function with `i` as image argument
4. DESCRIPTION
5. DESCRIPTION


## Glossary

`_` - this prefix means that this is difference between previous and current element

`_` - this postfix means that this is array 

`i` - variable, that represent image in two-dimensional array (gray scale image)

`i_h` - variable that represent input image height 

`i_w` - variable that represent input image width

`p_r` - variable that set number of pixels (leftward and upward) compared to input pixel

`a_m` - average range of comparison of current pixel with previous pixels (leftward and upward) in `p_r`

`a_c` - rate of relative match which equals to sum of value divide on sum of inputs `V/I`

`_vp_` - sum of value pattern arrays

`_dp_` - sum of difference pattern arrays

`vb_` - sum of value blobs

`db_` - sum of difference blobs

`vn_` - sum of value nets

`dn_` - sum of difference nets

`patterns_` - array of computed patterns


## Questions


