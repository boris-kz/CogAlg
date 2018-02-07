CogAlg
======

Full introduction: www.cognitivealgorithm.info

Intelligence is ability to predict and plan (self-predict), which can only be done by discovering and projecting patterns. This definition is well established: pattern recognition is a core of any IQ test. The problem is, there was no general and constructive definition of either pattern or recognition (similarity or match). So, I came up with my own definitions, which directly translate into algorithm introduced here.

For excellent popular introduction to cognition-as-prediction thesis see “On Intelligence” by Jeff Hawkins and “How to Create a Mind“ by Ray Kurzweil. But on a technical level, both of them and most current researchers implement pattern discovery via artificial neural networks, which operate in a very coarse statistical fashion.
Less coarse (more selective) are Capsule Networks, recently introduced by Geoffrey Hinton et al. But they are largely ad hock, still work-in-progress, and also use layers of CNN. Neither CNN nor CapsNet is theoretically derived. I outline my approach below and then compare it to ANN, biological NN, CapsNet, and clustering.

This content is published under the Creative Commons Attribution 4.0 International License. You are encouraged to republish and rewrite it in any way you see fit, as long as you provide attribution and a link.

.

OUTLINE OF MY APPROACH

.

Proposed algorithm is a clean design for deep learning: non-neuromorphic, sub-statistical, comparison-first.
It’s a hierarchical search for patterns, by cross-comparing inputs over incremental distance and composition. First-level inputs are single variables, such as pixels, and higher-level inputs are multi-parameter patterns formed on lower levels. Parameters (variables) include match and miss per variable compared on lower levels. Hence, higher-level patterns are more complex: variables per pattern selectively multiply on each level.

I define pattern as contiguous span of inputs forming the same sign of difference between input and feedback (part 2). There can be multiple orders of feedback, forming corresponding orders of overlapping patterns.
First-order feedback is prior input, and patterns are spans of inputs with increasing or decreasing magnitude.
Second-order feedback is average higher-level match, and second-order patterns are spans of inputs with above- or below- average match to prior inputs. And so on, with increasingly long-range feedback (if any) compared to higher order of match between input and shorter-range feedback.

Match and miss are formed by cross-comparing inputs, over selectively extended range of search. Basic comparison is inverse arithmetic operation between two single-variable inputs, starting with adjacent pixels.
Specific match and miss is determined by power of comparison: Boolean match is AND and miss is XOR, comparison by subtraction increases match to a smaller comparand and reduces miss to a difference,
comparison by division increases match to a multiple and reduces miss to a fraction, and so on (part 1).

To generalize, match is lossless compression per comparison, and match between patterns is combined match between their variables. To enable fine-grain selection of comparands, search expansion is strictly incremental. Thus, there is a unique set of operations per level, hence a singular in “cognitive algorithm“ (CogAlg below). Within level, search is incremental in distance between inputs and in their derivation order (part 4, level 1). Between levels, search is incremental in compositional scope and number of derived variables per pattern.

My hierarchy is a dynamically extended pipeline: when pattern terminates, it is outputted for comparison on the next level, and is replaced by initialized pattern on current level. Thus, a new level must be formed for a pattern terminated by current top level. This continues indefinitely, as long as system recieves new inputs.
As distinct from autoencoders (current mainstay in unsupervised learning), there is no need for decoding: comparison is done on each level, and level’s output patterns are also fed back to filter lower levels.

Autonomous cognition must start with analog inputs, such as video or audio. All symbolic data, including natural languages, is encoded by some cognitive process. To search for meaningful patterns, these symbols must be decoded before being cross-compared. And the difficulty of decoding is exponential with the level of encoding, so hierarchical learning starting with raw sensory input is by far the easiest to implement (part i).

To discover anything complex at “polynomial” cost, resulting patterns should also be hierarchical: each level of search adds a level of composition and a sub-level of differentiation to its input patterns. Higher-level search is selective per level of resulting pattern. Such composition and selection speeds up search, to form long-range spatio-temporal and then conceptual patterns. Which also send feedback: filters and then motor action, to select lower-level inputs and locations with above-average additive predictive value (part 3).

Hierarchical approaches are common in unsupervised learning, and all do some sort of pattern recognition.
But none that I know of is strictly incremental in scope and complexity of discoverable patterns. Incremental  selection is necessary for scalability, vs. combinatorial explosion in search space. But it’s more expensive upfront and won’t pay in simple test problems. So, it’s not suitable for immediate experimentation, which is probably why no one else seems to be working on anything sufficiently similar to my algorithm.

.

COMPARISON TO ARTIFICIAL AND BIOLOGICAL NEURAL NETWORKS

.

ANN learns via some version of Hebbian “fire together, wire together” coincidence reinforcement. Normally, “neuron’s” inputs are weighed at synapses, then summed and thresholded into output. Final output also back- propagates to synapses and is combined with their last input to adjust the weights. This weight adjustment is learning. But prior summation degrades initial resolution of inputs, precluding any comparison between them (the inputs are recoverable, but why degrade and then spend a ton of computation restoring them).

This is an inherently coarse statistical method: inputs are summed within samples defined by initially random weights. These weights are trained into meaningful values by SGD: practically a brute force and too expensive to scale without supervision or task-specific reinforcement. Again, CogAlg is comparison-first and resulting patterns are immediately meaningful. Basically, my feedback is infinitely finer-grained than backpropagation.

Currently the most successful method is CNN, which computes match as a product: input * kernel (weights), adjusted by activation function. But kernels are formed by incredibly stupid process, starting with random weights and adjusted by very coarse trial and error, delayed by hidden layers. The brain is born with a fixed number of neurons, they can’t be grown on demand and can only learn by growing and adjusting connections. But in software, generating new nodes to locally represent learned content should be far more efficient.

And similarity represented by weights is vastly exaggerated by matrix multiplication: conceptually, match is a common subset, not superset of comparands. Multiplication is analog (cheap) in the brain, and exaggeration adds robustness against noise, but at the cost of inherent inaccuracy. Which is partly compensated by some activation function, but that’s another unprincipled hack. I believe noise cancellation should be a separate function, specific to local properties of noise, which should be learned and updated along with the input itself.

ANN compute difference  (error) on their output layer, but not between hidden layers. This distinction is only justified in supervised learning, where we have some specific expectation. In unsupervised learning, all layers are equally unknown before input. So, each layer should compute both similarity (kernel) and difference (feedback) to lower layer. In temporally continuous input flow, such difference represents update. I think lack of explicit update values is what causes initialization bias in ANN and similar confirmation bias in humans.

Also, both input and kernel in ANN are arrays, often 2D or higher. This is far more coarse, thus potentially less selective and efficient, than one-to-one comparison of laterally adjacent inputs in my algorithm.
I use “level” vs. “layer” because my levels are not identical. Complexity of inputs and operations is incremental with elevation: initial inputs are pixels, higher-level inputs are patterns formed on lower levels. My inference is comparison that forms separate match and difference, and difference is fed back to update level-wide filters.

Inspiration by the brain kept ANN research going for decades before they became useful. But their “neurons” are mere stick figures for real ones. Of course, most of complexity in a neuron is due to constraints of biology. Ironically, weighted summation in ANN may also be a no-longer needed compensation for such constraint:
neural memory requires dedicated connections (synapses), which makes individual input representation and comparison prohibitively expensive. But not anymore, we now have dirt-cheap random access memory.

Other biological constraints are very slow neurons, and the imperative of fast reaction for survival in the wild. Both favor fast though crude summation (vs. slower one-to-one comparison), at the cost of glacial training. Reaction speed became less important: modern society is quite secure, while continuous learning is far more important because of accelerating progress. Another constraint is noise: neurons often fire at random, so their spikes are summed to reduce noise. Which is not a good reason to degrade far more precise electronic signals.

.

COMPARISON TO CAPSULE NETWORKS AND CLUSTERING

.

The nearest experimentally successful method is recently introduced “capsules”. Some similarities to CogAlg:
- capsules also output multi-variate vectors, “encapsulating” multiple properties, similar to my patterns
- these properties also include coordinates and dimensions, compared to compute differences and ratios
- these distances and proportions are also compared to find “equivariance” or affine transformations
- capsules also send direct feedback to lower layer (dynamic routing), vs. trans-hidden-layer backprop in ANN

But measure of similarity in CapsNet (“agreement” in dynamic routing) is still an unprincipled dot product, vs. conceptually consistent compression in CogAlg. This is critical, selection criterion is a core of my  algorithm. And CapsNet use CNN layers to recognize basic features, which are then fed to capsule layers. But in a truly general method, the same principles must apply on all stages of processing, any differention should be learned rather than built-in. All variables in my patterns are derived by incrementally complex comparison.

And current implementation of capsules contain only probability and pose variables. My patterns have match instead of probability, and include lower-level properties, with their matches and differences, as well as pose. In my terms, Hinton’s equivariance is a match between variables representing differences and distances. After being summed within pattern, these derivatives are evaluated to extend intra-pattern search and for feedback.
These are direct consequences of having comparison as a common atomic operation for all stages of CogAlg.

Also, my hierarchy is a dynamic pipeline: pattern is displaced from its level by a miss to new input, then forwarded to existing or newly formed higher level. Which means that higher-level patterns include lower-level variables, in addition to newly derived ones. So, both hierarchy of patterns per system and sub-hierarchy of variables per pattern expand with experience. This is fundamentally superior to static design of CapsNet.

Another method similar to mine is hierarchical clustering. But conventional clustering defines match as inverted difference between inputs. This is the opposite of ANN, which computes match but no difference. And it’s also wrong: match is a common subset of comparands, distinct from and complementary to the difference between them. Both should be computed because each has independent predictive value.

.

IMPLEMENTATION

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


