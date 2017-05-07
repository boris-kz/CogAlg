CogAlg
======

Full introduction: www.cognitivealgorithm.info

Proposed algorithm is a clean design for deep learning: non-neuromorphic, sub-statistical, comparison-first. It is derived from my definitions: pattern is a span of matching inputs and match is partial overlap between these inputs, defined by comparisons in hierarchically selective search. This search is strictly incremental, primarily in the distance between comparands and then in their derivation by prior comparisons. That means operations should be unique for each level of search and feedback, hence a singular in “cognitive algorithm“.

My outline is similar to hierarchical clustering, but match there is inverted difference between inputs. This is wrong, match is subset common for both comparands, distinct from and complementary to their difference. Also, neither deep ANN nor conventional hierarchical clustering implement incremental syntax (number of variables per input: parts 2 and 4) and incremental spatio-temporal dimensionality (parts 2 and 6).

Autonomous cognition starts by cross-comparing quantized analog (sensory) input, such as video or audio. All symbolic data in natural and artificial languages is a product of some prior cognitive process. Such data is implicitly encoded, and must be decoded before being searched for meaningful patterns. The difficulty of decoding is exponential with the level of encoding, so strictly bottom-up search is the easiest to implement. Hence, my initial inputs are pixels and higher-level inputs are patterns from lower-level comparison (part i).

I quantify match and miss by cross-comparing inputs, over selectively extended range of search (starting from adjacent pixels). Basic comparison is inverse arithmetic operation between two single-variable inputs. Specific measure of match and miss depends on power of such comparison: Boolean match is AND and miss is XOR, comparison by subtraction increases match to a smaller comparand and reduces miss to a difference, comparison by division increases match to a multiple and reduces miss to a fraction, and so on (part 1).

To discover anything complex at “polynomial” cost, both search and resulting patterns must be hierarchical: comparisons compress inputs into patterns, which are computationally cheaper to search on the next level. Comparison is also selective per level of differentiation (prior miss) within these hierarchically composed patterns. This discovery of compositional compression is iterated on incrementally higher levels, forming spatio-temporal and then conceptual patterns. Higher levels also send feedback: filters and then motor action, to select lower-level inputs and locations with above-average additive predictive value (part 3).

Hierarchical approaches are common in unsupervised learning, and all do some sort of pattern recognition. But none that I know of implement strictly incremental growth in scope and complexity of discoverable patterns. This is critical for scalability: each increment allows for predictive input selection, to avoid combinatorial explosion in search space. However, this selection is more expensive upfront and won’t pay off in simple problems: the usual test cases. Thus, my approach is not suitable for immediate experimentation, which is probably why no one else seems to be working on anything sufficiently similar.

Search space (above) is the number of patterns * number of their variables. Number of variables per pattern is incremental, starting from single-variable inputs, such as pixels in video. Each level of search adds new variables: match and miss, to every compared variable of an input pattern. So, the number of variables multiplies at every level, new variables are summed within a pattern, then combined to evaluate that pattern for expanded search. Feedback of averaged new variables extends lower-level algorithm by adding operations that filter future inputs, and then by adjusting the filters.

CogAlg implementation for image recognition: incremental-dimensionality pattern discovery (partial implementation of “incremental space-time dimensionality” section of part 2 in my core article):

level 1 compares consecutive 0D pixels within horizontal scan line, forming 1D patterns: line segments

level 2 compares contiguous 1D patterns between consecutive lines in a frame, forming 2D patterns: blobs

level 3 compares contiguous 2D patterns within a frame, forming discontinuous 2D patterns: groups of similar blobs

level 4 compares discontinuous 2D patterns between different images, forming groups of blobs that match across images

Initial testing will be done on labeled images, something like ImageNet, starting with level 4.

It can’t be done on lower levels because human vision is natively 2D, while my algorithm is incremental in dimensionality.

So, my lower levels are defined theoretically.


