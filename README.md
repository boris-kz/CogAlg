CogAlg
======

Full introduction: www.cognitivealgorithm.info

Proposed algorithm is a clean design for deep learning: non-neuromorphic, sub-statistical, comparison-first. It is derived from my definitions: pattern is a span of matching inputs and match is partial overlap between these inputs, defined by comparisons in hierarchically selective search. This search is strictly incremental, primarily in the distance between comparands and then in their derivation by prior comparisons. That means operations should be unique for each level of search and feedback, hence a singular in “cognitive algorithm“.

The above looks similar to hierarchical clustering, but it defines match as inverted difference between inputs. Which is wrong, match is subset common for both comparands, distinct from and complementary to their difference. Also, neither deep ANN nor conventional hierarchical clustering implements incremental syntax (number of variables per input) and incremental spatio-temporal dimensionality.

Autonomous cognition starts by cross-comparing quantized analog (sensory) input, such as video or audio. All symbolic data in natural and artificial languages is a product of some prior cognitive process. Such data is implicitly encoded, and must be decoded before being searched for meaningful patterns. The difficulty of decoding is exponential with the level of encoding, so strictly bottom-up search is the easiest to implement. Hence, my initial inputs are pixels and higher-level inputs are patterns from lower-level comparison.

I quantify match and miss by cross-comparing inputs, over selectively extended range of search (starting from adjacent pixels). Basic comparison is inverse arithmetic operation between two single-variable inputs. Specific measure of match and miss depends on power of such comparison: Boolean match is AND and miss is XOR, comparison by subtraction increases match to a smaller comparand and reduces miss to a difference, comparison by division increases match to a multiple and reduces miss to a fraction, and so on.

To discover anything complex at “polynomial” cost, both search and resulting patterns must be hierarchical: comparisons compress inputs into patterns, which are computationally cheaper to search on the next level. Comparison is also selective per level of differentiation (prior miss) within these hierarchically composed patterns. This discovery of compositional compression is iterated on incrementally higher levels, forming spatio-temporal and then conceptual patterns. Higher levels also send feedback: filters and then motor action, to select lower-level inputs and locations with above-average additive predictive value.

Hierarchical approaches are common in unsupervised learning, and all do some sort of pattern recognition. But none that I know of implement strictly incremental growth in scope and complexity of discoverable patterns. This is critical for scalability: each increment allows for predictive input selection, to avoid combinatorial explosion in search space. However, this selection is more expensive upfront and won’t pay off in simple problems: the usual test cases. Thus, my approach is not suitable for immediate experimentation, which is probably why no one else seems to be working on anything sufficiently similar.

Search space (above) is the number of patterns * number of their variables. Number of variables per pattern is incremental, starting from single-variable inputs, such as pixels in video. Each level of search adds new variables: match and miss, to every compared variable of an input pattern. So, the number of variables multiplies at every level, new variables are summed within a pattern, then combined to evaluate that pattern for expanded search. Feedback of averaged new variables extends lower-level algorithm by adding operations that filter future inputs, and then by adjusting the filters.

.

Any prediction has two components: what and where. We must have both: value of prediction = precision of what * precision of where. That “where” is currently neglected: statistical ML methods represent S-T dimensions with a significant lag, much more coarsely than the inputs themselves, Hence, precision of where (spans of and distances between patterns) is very low, and so is predictive value of combined representations. There is no such immediate degradation of positional information in my method.

My core algorithm is 1D: time only. Our space-time is 4D, but each of  these dimensions can be mapped on one level of search. This way, each level can form patterns that are strong enough to justify the cost of representing additional dimension and derivatives (matches and differences) in that dimension.
Again, initial inputs are pixels of video, or equivalent limit of positional resolution in other modalities.
Initial 4D cycle of search would compare contiguous inputs, analogously to connected-component analysis:

level 1 compares consecutive 0D pixels within horizontal scan line, forming 1D patterns: line segments.
level 2 compares contiguous 1D patterns between consecutive lines in a frame, forming 2D patterns: blobs.
level 3 compares contiguous 2D patterns between incremental-depth frames, forming 3D patterns: objects.
level 4 compares contiguous 3D patterns in temporal sequence, forming 4D patterns: processes.

Subsequent cycles would compare 4D input patterns over increasing distance in each dimension, forming longer-range discontinuous patterns. These cycles can be coded as implementation shortcut, or discovered by core algorithm itself, which should adapt to inputs of any dimensionality. “Dimension” here is a parameter that defines external sequence and distance among inputs. This is different from conventional clustering, which treats both external and internal parameters as dimensions.

However, average match in our space-time is presumably equal over all four dimensions. That means patterns defined in fewer dimensions will be biased to the angle of scanning, introducing artifacts. Hence, initial pixel comparison and inclusion into patterns should also be over 4D, or at least over 2D for images.

I am currently working on implementation of core algorithm to process images: level_1.py and level_2.py here,
and also on its natively-2D adaptation: level_1q.py here.
Initial testing will be recognition and automatic labeling of manually labeled images, from something like ImageNet.

This algorithm will be organically extended to process colors, then video, then stereo video (from multiple confocal cameras).
For video, level 3 will process consecutive frames and derive temporal patterns, and levels 4 and higher will process discontinuous 2D + time patterns. It should also extend to any type and scope of data.

Suggestions and collaboration is most welcome, see last part of my intro on prizes.


