'''   
    frame_blobs() defines blobs: contiguous areas of positive or negative deviation of gradient. Gradient is estimated 
    as |dx| + |dy|, then selectively and more precisely as hypot(dx, dy), from cross-comparison among adjacent pixels.

    Complemented by intra_blob (recursive search within blobs), it will be a 2D version of first-level core algorithm.
    frame_blobs() performs several levels (Le) of encoding, incremental per scan line defined by vertical coordinate y.
    value of y per Le line is shown relative to y of current input line, incremented by top-down scan of input image:

    1Le, line y:   comp_pixel (lateral and vertical comp) -> pixel + derivatives tuple: dert ) frame of derts: dert__ 
    2Le, line y-1: form_P(dert2) -> 1D pattern P
    3Le, line y-2: scan_P_(P, hP)-> hP, roots: down-connections, fork_: up-connections between Ps 
    4Le, line y-3: form_segment(hP, seg) -> seg: merge vertically-connected _Ps in non-forking blob segments
    5Le, line y-4+ seg depth: form_blob(seg, blob): merge connected segments in fork_ incomplete blobs, recursively  

    All 2D functions (y_comp, scan_P_, form_segment, form_blob) input two lines: higher and lower, convert elements of 
    lower line into elements of new higher line, then displace elements of old higher line into higher function.

    Higher-line elements include additional variables, derived while they were lower-line elements.
    Processing is mostly sequential because blobs are irregular and very difficult to map to matrices.

    prefix '_' denotes higher-line variable or pattern, vs. same-type lower-line variable or pattern,
    postfix '_' denotes array name, vs. same-name elements of that array.
'''