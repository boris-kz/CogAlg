Welcome to the CogAlg wiki!

Much of the coding was done by [Khanh Nguyen](https://github.com/khanh93vn/CogAlg) and [Kok Wei Chee](https://github.com/kwcckw/CogAlg), according to the principles introduced in README.

Main principle is input cross-comparison, then clustering by resulting match. This is done on all levels of composition from hierarchical clustering. First level cross-comp (cross-correlation) is among pixels, computing derivatives of brightness in a sliding kernel. Next step is segmentation: pixels + pixel-level derivatives ("derts") are clustered into positive and negative patterns: contiguous spans of above or below- average match.

To preserve positional info for evaluation, such algorithm must be specific to external (Cartesian) dimensionality of the input. 
Thus, we have three self-contained dimensionality-specific folders with separate workflow, explained below.
Beyond 1D, derivatives per dimension are combined into gradient. Cross-comparison here is very similar to basic edge-detection operator, resulting patterns are blobs of same-sign deviation of kernel-level match or gradient.

Design and exploration of this algorithm is done in the following sequence:


**line_1D_alg:**

- [line_Ps](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_P.py) (old stand-alone version: [line_patterns](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_patterns.py) is 1st-level core algorithm for 1D input: horizontal rows of pixels within an image. It forms patterns: line segments with same-sign deviation of difference between pixels, as well as sub-patterns by divisive hierarchical clustering. Sub-patterns are formed by recursive incremental range or incremental derivation cross-comp within selected patterns.  
- [line_PPs](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_PPs.py) is a 2nd-level 1D algorithm, mostly complete. It cross-compares each parameter of 1st-level patterns and forms Pps: param patterns, which in the aggregate represent patterns of patterns: PPs. It contains extended versions of 1st-level functions, as well as some new ones to handle intra-pattern hierarchy. 
- [line_recursive](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_recursive.py) is for level-recursion from 3rd level and higher. It forms incrementally higher levels of pattern composition and syntactic complexity, also by cross-comparison and agglomerative clustering. This should be final module of 1D algorithm, indefinitely scalable in the complexity of discoverable patterns, currently our main focus.

1D algorithm is a prototype, exclusively 1D patterns are not terribly informative / predictive in our 4D space-time. But it's best to develop basic principles and operations, which can then be converted to work in higher dimensions.


**frame_2D_alg:**

 1st level: [Chart](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/1st_level_2D_alg.png).
 Functional code:

- [frame_blobs](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/frame_blobs.py) defines initial blobs: contiguous areas of same-sign deviation of gradient per pixel. It's a basic cross-correlation and connectivity clustering within an image, but resulting blobs also contain comparison-derived parameters for future comparison between blobs (comp_blob). This code is functional. 
- [frame_blobs_par](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/frame_blobs_par.py) is POC of parallel-processing version. It is currently not much faster, but critical for potential scalability. 

- [intra_blob](https://github.com/boris-kz/CogAlg/tree/master/frame_2D_alg/intra_blob): recursive calls to intra_comp: cross-comparison at extended range or higher derivation within selected blobs and then sub_blobs. Each call converts input blob into root blob and forms respective sub_blobs: contiguous areas of same-sign deviation of a new gradient.

  [Diagram](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_scheme.png), 
   
  - [intra_comp](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/intra_comp.py) cross-compares pixels over extended range, or cross-compares angle of gradient, forming corresponding type of new gradient.
  - [draw_intra_comp](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/draw_intra_comp.py) computes gradients of the first three layers of forks over the whole frame, for visualization only (old version).
    
  - [comp_slice_flip](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/comp_slice_flip.py) will be terminal fork of intra_blob, selective for smooth elongated high-gradient blobs: likely edges or contours. It forms edge-orthogonal Ps: horizontal blob slices, then cross-compares vertically consecutive Ps. This is a 2D -> 1D dimensionality reduction, converting edges into vector representations. It is similar to the second level of 1D alg, which cross-compares horizontally discontinuous Ps: [line_PPs_draft](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_PPs_draft.py). 

    Root function of comp_slice_flip is slice_blob, that and other other functions down to comp_slice are functional. Comp_slice and below is a work in progress.
  
 2nd level and a prototype for recursive meta-level 2D algorithm, work in progress:
 
   - [comp_blob_draft](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/comp_blob_draft.py): cross-comp of blobs formed by cross-comp of the same range and derivation within root blob or frame. 
     Cross-comp is default for top layer of each blob, with specification by cross-comp of deeper layers if input value + match > average. Clustering the blobs (by match only?) forms incrementally higher-composition super-blobs.
   - comp_layer_: cluster | reorder -> bi-directional hierarchy? sub_blobs comp to higher-blob: contour or axis?
     (called from comp_blob) 
   - eval_overlap: redundant reps of combined-core positive blob areas, vertical or cross-fork. 
    
  
**video_3D_alg:**

- [video_draft()](https://github.com/boris-kz/CogAlg/blob/master/video_3D_alg/video_draft.py) is obsolete 3D extension of pixel-level cross-correlation, as in frame_blobs. Eventually, it will extend all of 2D alg with time dimension. This version will be made fully recursive and hopefully effective in real world.


Higher levels of each D-cycle algorithm will do discontinuous search among dimensionally-complete patterns: blobs or graphs. The most complex part there will be cross-comparison, because it has to operate on multi-level, multi-variate input patterns, with multiple arithmetic powers (such as comparison by division). And each comparison will have variable and generally expanding range, resulting in looser proximity clustering, vs. simple connectivity clustering on the 1st level. Discontinuity in clustering results from that in cross-comp. 

Complete hierarchical algorithm will have two-level code: 

- 1st level algorithm: contiguous cross-comparison and clustering over full-dimensional cycle (up to 4D), plus feedback to adjust most and least significant bits of the input. 
- Recurrent increment in complexity, extending current-level alg to next-level alg. This increment will account for increasing internal complexity of input patterns on higher levels, unfolding them for cross-comparison, then re-combining the results for evaluation and feedback of average deviation of each derivative.

We will then add colors, maybe audio and text. Initial testing could be recognition of labeled images, but 2D is also a poor representation of our 4D world, we will probably start directly with video or stereo video.

Suggestions and collaboration are most welcome, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).