CogAlg
======

Full introduction: www.cognitivealgorithm.info

Proposed algorithm is a clean design for deep learning: non-neuromorphic, sub-statistical, comparison-first.
It’s a hierarchical search for patterns, similar to hierarchical clustering. But conventional clustering defines match as inverted difference between inputs, which is wrong. Match is a subset common for both comparands, distinct from and complementary to their difference. Also, I don’t know of any clustering that implements incremental syntax: number of variables per input, and incremental spatio-temporal dimensionality (part 2).

I define pattern as a span of matching inputs, and match as partial overlap between them. Search should be strictly incremental, in distance between inputs and their derivation within a level of search (part 4, level 1), and then in compositional scope and derivation of patterns formed on lower levels. “Strictly incremental” means that there must be a unique set of operations per level, hence a singular in “cognitive algorithm“.

Autonomous cognition must start with analog inputs, such as video or audio. All symbolic data, including natural languages, is encoded by some cognitive process. Thus, it should be decoded before being cross- compared in search for meaningful patterns. And the difficulty of decoding is exponential with the level of encoding, so search that starts with raw sensory input is by far the easiest to implement (part i).
Hence, my initial inputs are pixels, and higher-level inputs are patterns formed by lower-level search.

I quantify match and miss by cross-comparing inputs over selectively extended range of search. Basic comparison is inverse arithmetic operation between two single-variable inputs, starting with adjacent pixels.
Specific match and miss is determined by power of comparison: Boolean match is AND and miss is XOR, comparison by subtraction increases match to a smaller comparand and reduces miss to a difference,
comparison by division increases match to a multiple and reduces miss to a fraction, and so on (part 1).

To discover anything complex at “polynomial” cost, both search and resulting patterns must be hierarchical:
comparisons group inputs into patterns, which are compared on the next level. Comparison is also selective per level of differentiation (prior miss) within these hierarchically composed patterns. Such compression and selection reduces search time, relative to comparing original inputs. Higher levels form increasingly long-range spatio-temporal and then conceptual patterns. They also send feedback: filters and then motor action, to select lower-level inputs and locations with above-average additive predictive value (part 3).

Hierarchical approaches are common in unsupervised learning, and all do some sort of pattern recognition. But none that I know of implement strictly incremental growth in scope and complexity of discoverable patterns. This is critical for scalability: each increment allows for predictive input selection, to avoid combinatorial explosion in search space. However, this selection is more expensive upfront and won’t pay in simple problems, which are the usual test cases. Thus, my approach is not suitable for immediate experimentation, which is probably why no one else seems to be working on anything sufficiently similar.

This content is published under the Creative Commons Attribution 4.0 International License. You are encouraged to republish and rewrite it in any way you see fit, as long as you provide attribution and a link.

.

Any prediction has two components: what and where. We must have both: value of prediction = precision of what * precision of where. That “where” is currently neglected: statistical ML methods represent S-T dimensions with a significant lag, much more coarsely than the inputs themselves, Hence, precision of where (spans of and distances between patterns) is severely degraded, and so is predictive value of combined representations. There is no such immediate degradation of positional information in my method.

My core algorithm is 1D: time only (part 4). Our space-time is 4D, but each of these dimensions can be mapped on one level of search. This way, levels can select input patterns that are strong enough to justify the cost of representing additional dimension, as well as derivatives (matches and differences) in that dimension.
Initial 4D cycle of search would compare contiguous inputs, analogously to connected-component analysis:

level 1 compares consecutive 0D pixels within horizontal scan line, forming 1D patterns: line segments.

level 2 compares contiguous 1D patterns between consecutive lines in a frame, forming 2D patterns: blobs.

level 3 compares contiguous 2D patterns between incremental-depth frames, forming 3D patterns: objects.

level 4 compares contiguous 3D patterns in temporal sequence, forming 4D patterns: processes.

Subsequent cycles would compare 4D input patterns over increasing distance in each dimension, forming longer-range discontinuous patterns. These cycles can be coded as implementation shortcut, or discovered by core algorithm itself, which can adapt to inputs of any dimensionality. “Dimension” here is a parameter that defines external sequence and distance among inputs. This is different from conventional clustering, which treats both external and internal parameters as dimensions. More in part 6.

However, average match in our space-time is presumably equal over all four dimensions. That means patterns defined in fewer dimensions will be biased by the angle of scanning, introducing artifacts. Hence, initial pixel comparison and inclusion into patterns should also be over 4D, or at least over 2D for images.
This is a universe-specific extension of core algorithm.

I am currently working on implementation of core algorithm to process images: level_1_working.py and level_2_old.py here, and also on its natively-2D adaptation: level_1_2D_draft.py here.
Initial testing will be recognition and automatic labeling of manually labeled images, from something like ImageNet.

This algorithm will be organically extended to process colors, then video, then stereo video (from multiple confocal cameras).
For video, level 3 will process consecutive frames and derive temporal patterns, and levels 4 and higher will process discontinuous 2D + time patterns. It should also extend to any type and scope of data.

Suggestions and collaboration are most welcome, see last part of my intro on prizes.


