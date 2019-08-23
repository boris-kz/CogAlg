Welcome to the CogAlg wiki!

Most of the programming is currently done by [Khanh Nguyen](https://github.com/khanh93vn/CogAlg), according to the principles introduced in README. I edit and paste his updates here for an overview, but his repository is best for running code. The code is divided into three self-contained folders:

**line_1D_alg:**

- [line_POC](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_POC.py) is 1st-level core algorithm, to demonstrate basic principles. It works, but 1D patterns are not effective in recognition and prediction in our 4D space-time.

**frame_2D_alg:** currently a work-in-progress, will process still images:

- [frame_blobs](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/frame_blobs.py) defines initial blobs: contiguous areas of same-sign deviation of gradient per pixel. This is a basic clustering / segmentation within an image, but resulting blobs contain comparison-derived parameters for future comparison between blobs. This code is functional.

- [intra_blob](https://github.com/boris-kz/CogAlg/tree/master/frame_2D_alg/intra_blob): conditional recursively extended search within selected blobs and then sub_blobs, which converts them into master blob and respective sub_blobs:

  - intra_comp is called by intra_blob to perform comparison over extended range or higher derivation, and form corresponding master blob. Each intra-blob() may call two layers of intra_comp(): first for comparison over input parameter, then for comparison over angle of gradient derived by the first comparison. Both define corresponding sub_blobs: contiguous areas of same-sign deviation of resulting gradient.
   
    - [comp_i](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/comp_i.py) compares input parameter, or computes and compares angle of resulting gradient, to compute corresponding input gradient or angle gradient, over defined range.
    
    - [comp_i_frame](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/comp_i_frame.py) computes and displays gradients of the first three layers of forks over the whole frame.
    This version is only for debugging, it doesn't form blobs.
    
  - [comp_P_draft](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/comp_P_draft.py): a draft for comparison between vertically consecutive Ps: horizontal slices of blob segments. This will be similar to higher levels of 1D alg.

- comp_blob:

  higher levels of 2D alg will compare between blobs within image, forming incrementally higher-composition super-blobs. There is no code on that yet.
  
**video_3D_alg:**

- [video_draft()](https://github.com/boris-kz/CogAlg/blob/master/video_3D_alg/video_draft.py) is obsolete 3D extension of frame_blobs. Eventually, it will extend all of 2D alg with time dimension. This version will hopefully be fully recursive and effective in real world.

Higher levels for each D-cycle alg will process discontinuous search among full-D patterns.
Complete hierarchical algorithm will have two-level code: 
- 1st level algorithm: contiguous cross-comparison over full-D cycle, plus feedback to adjust most and least significant bits of the input. 
- Recurrent increment in complexity, extending current-level alg to next-level alg. This increment will account for increasing internal complexity of input patterns on higher levels, unfolding them for cross-comparison and re-folding results for evaluation and feedback. 

We will then add colors, maybe audio and text. Initial testing could be recognition of labeled images, but 2D is a poor representation of our 4D world, we will probably start directly with video or stereo video.

Suggestions and collaboration are most welcome, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).