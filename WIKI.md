Welcome to the CogAlg wiki!

Much of the coding was done by [Khanh Nguyen](https://github.com/khanh93vn/CogAlg) and [Kok Wei Chee](https://github.com/kwcckw/CogAlg), according to the principles introduced in README.

This project is supposed to be conceptually consistent implementation of hierarchical clustering, both divisive and agglomerative. Each level performs input cross-comparison, then clustering by resulting match or difference. 
First level is basically edge-detection: cross-comp among pixels in a sliding kernel, followed by flood-fill: image segmentation by the sign of resulting match (inverse deviation of gradient).

Higher-level cross-comp is over increasing Euclidean distance between input elements, which are parameterized clusters (patterns) formed on the lower level. Thus, multi-level clustering forms hierarchical graphs, with incremental composition.
Recursively higher levels are composed of older clusters that were pushed out of lower level because they got beyond comparison range for new inputs. Such hierarchy is a pipeline, where each level is also a pipeline.

Unique features of this project:
- higher-order match (numeric similarity) is shared quantity, initially min comparands, a measure of primary compression, 
- value of miss / differences is borrowed from co-derived or co-projected match, because they form competing predictions,
- higher-level representations are encoded as derivatives: match and miss per lower-level variable or cluster,
- representation of external coordinates and dimensions: predictive value = precision of what * precision of where.

That means algorithm must be specific to external dimensionality of an input vector, initially defined in Cartesian coordinates. Which is why we currently have three self-contained dimensionality-specific folders with separate workflow. Exploration and design of this algorithm is done with incremental dimensionality:

**line_1D_alg:**

- [line_Ps](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_P.py) (old stand-alone version: [line_patterns](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_patterns.py)) is a core 1st-level algorithm for 1D input: horizontal rows of pixels within an image. It forms patterns: line segments with same-sign deviation of difference between pixels, as well as sub-patterns by divisive hierarchical clustering. Sub-patterns are formed by recursive incremental range or incremental derivation cross-comp within selected patterns.  
- [line_PPs](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_PPs.py) is a 2nd-level 1D algorithm, mostly done. It cross-compares each parameter of 1st-level patterns and forms Pps: param patterns, which in the aggregate represent patterns of patterns: PPs. It contains extended versions of 1st-level functions, as well as some new ones to handle intra-pattern hierarchy. 
- [line_recursive](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_recursive.py) is for level-recursion from 3rd level and higher. It forms incrementally higher levels of pattern composition and syntactic complexity, also by cross-comparison and agglomerative clustering. This should be final a module of 1D algorithm, indefinitely scalable in the complexity of discoverable patterns, still tentative and out of date.

1D algorithm is a prototype, exclusively 1D patterns are not terribly informative / predictive in our 4D space-time. 
But it's the easiest to develop basic principles and operations, which can then be extended to work in higher dimensions.


**frame_2D_alg:**

 1st level: [Chart](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/1st_level_2D_alg.png).
 Functional code: 

- [frame_blobs](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/frame_blobs.py) defines initial blobs: contiguous areas of same-sign deviation of gradient per pixel. It's a basic cross-correlation and connectivity clustering within an image, but resulting blobs also contain comparison-derived parameters for future comparison between blobs (comp_blob).
- [draw_frame_blobs](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/draw_frame_blobs.py)
- [frame_blobs_par](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/frame_blobs_par.py): old POC of parallel-processing version. It is currently not faster, but may be critical for potential scalability. 

  - [intra_blob](https://github.com/boris-kz/CogAlg/tree/master/frame_2D_alg/intra_blob): recursive calls to intra_comp: cross-comparison at extended range or higher derivation within selected blobs and then sub_blobs. Each call converts input blob into root blob and forms respective sub_blobs: contiguous areas of same-sign deviation of a new gradient. 
   [Diagram](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_scheme.png).
   
    - [intra_comp](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/intra_comp.py) cross-compares pixels or angle of gradient or gradient itself over extended range, forming corresponding type of new gradient.
    - [draw_intra_comp](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/draw_intra_comp.py) computes gradients of the first three layers of forks over the whole frame, for visualization only (old version).
  
    Work-in-progress:
    - [vectorize_edge_blob](https://github.com/boris-kz/CogAlg/tree/master/frame_2D_alg/vectorize_edge_blob) is a terminal fork of intra_blob, selective for smooth elongated high-gradient blobs: likely edges / contours of flat low-gradient blobs. Combination of connected edge blobs and flat blobs should represent visual objects. [vectorize_root](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/vectorize_edge_blob/root.py) calls:
    - slice_blob, which forms edge-orthogonal Ps: 1D patterns or blob slices.
    - [comp_slice](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/vectorize_edge_blob/comp_slice.py) cross-compares vertically consecutive Ps and forms PPs (2D patterns of patterns) along the axis of edge blob. 
    - [sub_recursion](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/vectorize_edge_blob/sub_recursion.py) is an extension of comp_slice for divisive clustering, cross-comparing Ps within exceptionally strong PPs at incremental distance (rng+) or derivation (der+), and combines them into sub-PPs, sub-sub-PPs, etc.
    - [agg_recursion](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/vectorize_edge_blob/agg_recursion.py) is an extension of comp_slice for agglomerative clustering. It cross-compares distant PPs, graphs of PPs, graphs of graphs, etc., along blob edges or skeletons. Resulting graphs (Gs) will have incremental composition and sparsity, still restricted to the edges of the same blob or merged blob. 
    
    This is a 2D -> 1D dimensionality reduction, converting edges into vector representations. It's a 2D analog of line_PPs and line_recursive in 1D alg, which cross-compares horizontally discontinuous Ps. The complexity of the code may seem like a huge overkill, but automatic edge tracing and vectorization is still an unsolved problem. More importantly, it's a prototype for general agglomerative recursion in 2D and higher, starting with frame_graphs.
  
 Initial drafts, way out of date:
 2nd level is [frame_graphs](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/frame_graphs.py): cross-comp of blobs formed by frame_blobs, forming graphs of blobs, currently a draft.
 Final module will be [frame_recursive](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/frame_recursive.py): compositionally recursive agglomerative clustering, a 2D analog of line_recursive in 1D. It will cross-compare top layer of each input blob, of whatever composition order, with conditional specification by cross-comp of deeper layers if input value + match > average. Clustering the blobs and then their graphs will form incrementally higher-composition graphs of blobs, graphs of graphs, etc.

  
**video_3D_alg:**

- [video_draft()](https://github.com/boris-kz/CogAlg/blob/master/video_3D_alg/video_draft.py): grossly obsolete stab at 3D extension of pixel-level cross-correlation, as in frame_blobs. Eventually, it will extend all of 2D alg with time dimension. This version will be made fully recursive and hopefully effective in real world.

Each D-cycle algorithm will have two-level code: 

- 1st level algorithm: contiguous cross-comparison and clustering over full-dimensional cycle (initially up to 4D), divisive recursion, and feedback to adjust dynamic range: most and least significant bits of the input. 
- Agglomerative and divisive recursion, forming incrementally higher and lower composition levels. Cross-comparison will operate on multi-level, multi-variate input patterns, with indefinite nesting and arithmetic powers of comparison, such as comparison by division. Cross-comp in divisive clustering will be over incremental comparison range (discontinuity) and derivation (differences of differences, etc). Comparison results will be combined for pattern evaluation and feedback of average deviation per derivative, adjusting filters (hyperparameters). 

We will then add colors, maybe audio and text. Initial testing could be recognition of labeled images, but 2D is also a poor representation of our 4D world, we will probably start directly with video or stereo video.
Ultimately, the algorithm should be able to operate on its own code, similar to how we work with pure math. That will be "self-improvement", including automatic generation of higher-dimensional algorithms, for space-time and derived dimensions, as well as operations on new input modalities.  

Suggestions and collaboration are most welcome, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).