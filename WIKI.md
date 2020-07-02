Welcome to the CogAlg wiki!

Much of the coding was done by [Khanh Nguyen](https://github.com/khanh93vn/CogAlg) and [Kok Wei Chee](https://github.com/kwcckw/CogAlg), according to the principles introduced in README.
We have three self-contained folders:

**line_1D_alg:**

- [line_patterns](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_patterns.py) is 1st-level core algorithm for 1D input: horizontal rows of pixels within an image, to demonstrate basic principles. But 1D patterns are not terribly informative / predictive in our 4D space-time.

**frame_2D_alg:** currently a work-in-progress, will process still images:

 1st level:
- [frame_blobs](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/frame_blobs.py) defines initial blobs: contiguous areas of same-sign deviation of gradient per pixel. It's a basic cross-correlation and connectivity clustering within an image, but resulting blobs also contain comparison-derived parameters for future comparison between blobs, which I haven't seen elsewhere. This code is functional.

- [intra_blob](https://github.com/boris-kz/CogAlg/tree/master/frame_2D_alg/intra_blob): Recursive calls to cluster_eval() cross-comparison (intra_comp) at extended range or higher derivation within selected blobs and then sub_blobs.  convert input blob into root blob and form respective sub_blobs: contiguous areas of same-sign deviation of new gradient.
   
  - [intra_comp](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/intra_comp.py) cross-compares input parameter over extended range and forms new gradient.
  - [draw_intra_comp](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/draw_intra_comp.py) computes gradients of the first three layers of forks over the whole frame, for visualization only.
  
  [comp_P_draft](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/comp_P_draft.py): pseudo code for additional fork that cross-compares between vertically consecutive Ps: horizontal slices of blob segments. This will be selective for highly elongated blobs: likely edge contours. It will perform a version line-tracing: 2D -> 1D dimensionality reduction, converting such blobs into vector representations. This is similar to higher levels of 1D alg (not implemented in line_patterns).
  
 2nd level and a prototype for recursive meta-level 2D algorithm, to be added:
 
   - merge_blob_: merge weak blobs (with negative summed value) into infra-blob, for comp_blob_ but not intra_blob,
   - comp_blob_: cross-comp of same range and derivation blobs within root blob ) frame, 
    forms incrementally higher-composition super-blobs, with selective extended cross-comp of their elements,
   - comp_layer_: cluster | reorder -> bi-directional hierarchy? sub_blobs comp to higher-blob: contour or axis? 
   - eval_overlap: redundant reps of combined-core positive blob areas, vertical or cross-fork? 
    
  
**video_3D_alg:**

- [video_draft()](https://github.com/boris-kz/CogAlg/blob/master/video_3D_alg/video_draft.py) is obsolete 3D extension of pixel-level cross-correlation, as in frame_blobs. Eventually, it will extend all of 2D alg with time dimension. This version will be made fully recursive and hopefully effective in real world.


Higher levels for each D-cycle alg will process discontinuous search among dimensionally-complete patterns.
Complete hierarchical algorithm will have two-level code: 
- 1st level algorithm: contiguous cross-comparison over full-D cycle, plus feedback to adjust most and least significant bits of the input. 
- Recurrent increment in complexity, extending current-level alg to next-level alg. This increment will account for increasing internal complexity of input patterns on higher levels, unfolding them for cross-comparison and re-combining the results for evaluation and feedback.

We will then add colors, maybe audio and text. Initial testing could be recognition of labeled images, but 2D is a poor representation of our 4D world, we will probably start directly with video or stereo video.

Suggestions and collaboration are most welcome, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).