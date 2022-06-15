"""
  line_Ps is a principal version of 1st-level 1D algorithm
  Operations:
  -
- Cross-compare consecutive pixels within each row of image, forming dert_: queue of derts, each a tuple of derivatives per pixel.
  dert_ is then segmented into patterns Pms and Pds: contiguous sequences of pixels forming same-sign match or difference.
  Initial match is inverse deviation of variation: m = ave_|d| - |d|,
  rather than a minimum for directly defined match: albedo of an object doesn't correlate with its predictive value.
  -
- Match patterns Pms are spans of inputs forming same-sign match. Positive Pms contain high-match pixels, which are likely
  to match more distant pixels. Thus, positive Pms are evaluated for cross-comp of pixels over incremented range.
  -
- Difference patterns Pds are spans of inputs forming same-sign ds. d sign match is a precondition for d match, so only
  same-sign spans (Pds) are evaluated for cross-comp of constituent differences, which forms higher derivatives.
  (d match = min: rng+ comp value: predictive value of difference is proportional to its magnitude, although inversely so)
  -
  Both extended cross-comp forks are recursive: resulting sub-patterns are evaluated for deeper cross-comp, same as top patterns.
  These forks here are exclusive per P to avoid redundancy, but they do overlap in line_patterns_olp.
"""

using Images, ImageView, CSV

# instead of the class in Python version in Julia values are stared in the struct
mutable struct Cdert_
    i::Int16
    p::Int16
    d::Int16
    m::Int16
    mrdn::Bool
end

dert_ = Cdert_[] # line-wide i_, p_, d_, m_, mrdn_


verbose = false
# pattern filters or hyper-parameters: eventually from higher-level feedback, initialized here as constants:
ave = 15  # |difference| between pixels that coincides with average value of Pm
ave_min = 2  # for m defined as min |d|: smaller?
ave_M = 20  # min M for initial incremental-range comparison(t_), higher cost than der_comp?
ave_D = 5  # min |D| for initial incremental-derivation comparison(d_)
ave_nP = 5  # average number of sub_Ps in P, to estimate intra-costs? ave_rdn_inc = 1 + 1 / ave_nP # 1.2
ave_rdm = 0.5  # obsolete: average dm / m, to project bi_m = m * 1.5
ave_splice = 50  # to merge a kernel of 3 adjacent Ps
init_y = 501  # starting row, set 0 for the whole frame, mostly not needed 
halt_y = 501  # ending row, set 999999999 for arbitrary image.
#! these values will be one more (+1) in the Julia version because of the numbering specifics
"""
    Conventions:
    postfix 't' denotes tuple, multiple ts is a nested tuple
    postfix '_' denotes array name, vs. same-name elements
    prefix '_'  denotes prior of two same-name variables
    prefix 'f'  denotes flag
    1-3 letter names are normally scalars, except for P and similar classes, 
    capitalized variables are normally summed small-case variables,
    longer names are normally classes
"""


function line_comp(pixel_)  
    local _i = pixel_[1]  #! differs from the python version
    # cross_comparison:
    for i in pixel_[2:end]  # pixel i is compared to prior pixel _i in a row:
        d = i - _i  # accum in rng
        p = i + _i  # accum in rng
        m = ave - abs(d)  # for consistency with deriv_comp output, else redundant
        mrdn = m < 0  # 1 if abs(d) is stronger than m, redundant here
        push!(dert_, Cdert_(i, p, d, m, mrdn))  # save data in the struct
        _i = i

    end
end


render = 0
fline_PPs = 0
frecursive = 0
logging = 1  # logging of local functions variables

image_path = "./line_1D_alg/raccoon.jpg";
# image_path = "/home/alex/Python/CogAlg/line_1D_alg/raccoon.jpg";
image = nothing

# check if image exist
if isfile(image_path)
    image = load(image_path)  # read as N0f8 (normed 0...1) type array of cells
else
    println("ERROR: Image not found!")
end

gray_image = Gray.(image)  # convert rgb N0f8 to gray N0f8 array of cells
img_channel_view = channelview(gray_image)  # transform to the array of N0f8 numbers
gray_image_int = convert.(Int16, trunc.(img_channel_view .* 255))  # finally get array of 0...255 numbers
# imshow(gray_image)

# Main
Y, X = size(gray_image) # Y: frame height, X: frame width
frame = []

# y is index of new row pixel_, we only need one row, use init_y=1, halt_y=Y for full frame
for y = init_y:min(halt_y, Y)
    line_Ps_root(gray_image_int[y, :])  # line = [Pm_, Pd_]
end

if logging == 1
    CSV.write("./line_1D_alg/layer0_log_jl.csv", dert_)
end
