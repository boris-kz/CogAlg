Welcome to the CogAlg wiki!

Much of the coding was done by [Khanh Nguyen](https://github.com/khanh93vn/CogAlg) and [Kok Wei Chee](https://github.com/kwcckw/CogAlg), according to the principles introduced in README.

This scheme is supposed to be conceptually consistent implementation of hierarchical clustering, both divisive and agglomerative. Each level performs input cross-comparison, then clustering by resulting match. 
First level is basically edge-detection: cross-comp among pixels in a sliding kernel, followed by flood-fill: image segmentation by the sign of resulting match (inverse deviation of gradient).

Higher-level cross-comp is over increasing distance between input elements, where each element is a parameterized cluster (pattern) formed on the lower level. Thus, multi-level clustering forms hierarchical graphs, with incremental composition.
Higher levels should be added recursively, composed of older clusters that were pushed out of lower level because they got beyond comparison range for new inputs. Such hierarchy is a pipeline, where each level is also a pipeline.

A key feature here is explicit coordinates, because input position is just as important as content: predictive value = precision of what * precision of where. That means algorithm must be specific to external dimensionality of the input vector, initially in Cartesian coordinates. Which is why we have three self-contained dimensionality-specific folders with separate workflow. Exploration and design of this algorithm is done with incremental dimensionality:

**line_1D_alg:**

- [line_Ps](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_P.py) (old stand-alone version: [line_patterns](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_patterns.py)) is a core 1st-level algorithm for 1D input: horizontal rows of pixels within an image. It forms patterns: line segments with same-sign deviation of difference between pixels, as well as sub-patterns by divisive hierarchical clustering. Sub-patterns are formed by recursive incremental range or incremental derivation cross-comp within selected patterns.  
- [line_PPs](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_PPs.py) is a 2nd-level 1D algorithm, mostly done. It cross-compares each parameter of 1st-level patterns and forms Pps: param patterns, which in the aggregate represent patterns of patterns: PPs. It contains extended versions of 1st-level functions, as well as some new ones to handle intra-pattern hierarchy. 
- [line_recursive](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_recursive.py) is for level-recursion from 3rd level and higher. It forms incrementally higher levels of pattern composition and syntactic complexity, also by cross-comparison and agglomerative clustering. This should be final a module of 1D algorithm, indefinitely scalable in the complexity of discoverable patterns, still tentative and out of date.

1D algorithm is a prototype, exclusively 1D patterns are not terribly informative / predictive in our 4D space-time. But it's best to develop basic principles and operations, which can then be extended to work in higher dimensions.


**frame_2D_alg:**

 1st level: [Chart](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/1st_level_2D_alg.png).
 Functional code: 

- [frame_blobs](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/frame_blobs.py) defines initial blobs: contiguous areas of same-sign deviation of gradient per pixel. It's a basic cross-correlation and connectivity clustering within an image, but resulting blobs also contain comparison-derived parameters for future comparison between blobs (comp_blob). 
- [frame_blobs_par](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/frame_blobs_par.py) is POC of parallel-processing version. It is currently not much faster, but critical for potential scalability. 

  - [intra_blob](https://github.com/boris-kz/CogAlg/tree/master/frame_2D_alg/intra_blob): recursive calls to intra_comp: cross-comparison at extended range or higher derivation within selected blobs and then sub_blobs. Each call converts input blob into root blob and forms respective sub_blobs: contiguous areas of same-sign deviation of a new gradient. 
   [Diagram](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_scheme.png).
   
    - [intra_comp](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/intra_comp.py) cross-compares pixels or angle of gradient or gradient itself over extended range, forming corresponding type of new gradient.
    - [draw_intra_comp](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/draw_intra_comp.py) computes gradients of the first three layers of forks over the whole frame, for visualization only (old version).
  
    Work-in-progress:
  
    - [comp_slice](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/comp_slice.py) is a terminal fork of intra_blob, selective for smooth elongated high-gradient blobs: likely edges or contours. It forms edge-orthogonal Ps: horizontal blob slices, then cross-compares vertically consecutive Ps to form PPs (patterns of patterns) along the edge. This is a 2D -> 1D dimensionality reduction, converting edges into vector representations. It is similar to the second level of 1D alg, which cross-compares horizontally discontinuous Ps: [line_PPs](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_PPs.py). 
    - [agg_recursion](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/agg_recursion.py) is an extension of comp_slice for agglomerative recursion. It cross-compares distant PPs, and then graphs of PPs, along blob edges or skeletons. Resulting graphs (Gs) will have incremental composition and sparsity, still restricted to edges of the same blob. The complexity of the code may seem like a huge overkill, but automatic edge tracing and vectorization is still an unsolved problem. And this is a prototype for general agglomerative recursion in 2D and higher.
  
 Initial drafts, way out of date:
 2nd level is [frame_bblobs](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/frame_bblobs.py): cross-comp of blobs formed by frame_blobs, forming higher-composition bblobs, currently a draft.
 Final module will be [frame_recursive](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/frame_recursive.py): compositionally recursive agglomerative clustering, a 2D analog of line_recursive in 1D. It will cross-compare top layer of each input blob, of whatever composition order, with conditional specification by cross-comp of deeper layers if input value + match > average. Clustering the blobs (by match only?) forms incrementally higher-composition super-blobs: bblobs, bbblobs, etc.

  
**video_3D_alg:**

- [video_draft()](https://github.com/boris-kz/CogAlg/blob/master/video_3D_alg/video_draft.py) is a grossly obsolete 3D extension of pixel-level cross-correlation, as in frame_blobs. Eventually, it will extend all of 2D alg with time dimension. This version will be made fully recursive and hopefully effective in real world.

Each D-cycle algorithm will have two-level code: 

- 1st level algorithm: contiguous cross-comparison and clustering over full-dimensional cycle (initially up to 4D), divisive recursion, and feedback to adjust dynamic range: most and least significant bits of the input. 
- Agglomerative recursion, forming incrementally higher-composition levels. Cross-comparison will operate on multi-level, multi-variate input patterns, with indefinite nesting and multiple arithmetic powers (such as comparison by division). And it will be over variable and generally expanding comparison range, resulting in looser proximity clustering, vs. simple connectivity clustering on the 1st level. Discontinuity within clusters is a consequence of increasing max distance between cross-compared inputs. Comparison results will be combined for pattern evaluation and feedback of average deviation per derivative, adjusting filters (hyperparameters). 

We will then add colors, maybe audio and text. Initial testing could be recognition of labeled images, but 2D is also a poor representation of our 4D world, we will probably start directly with video or stereo video.
Ultimately, the algorithm should be able to operate on its own code, similar to how we work with pure math. That will be "self-improvement", including automatic generation of higher-dimensional algorithms, for space-time and derived dimensions, as well as operations on new input modalities.  

Suggestions and collaboration are most welcome, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).