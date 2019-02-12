CogAlg
======

Full introduction: www.cognitivealgorithm.info

Intelligence is ability to predict and plan (self-predict), which can only be done by discovering and projecting patterns. This perspective is well established: pattern recognition is a core of any IQ test.
But there is no general and at the same time constructive definition of either pattern or recognition (quantified similarity) So, I came up with my own definitions, which directly translate into algorithm proposed below.

For excellent popular introductions to cognition-as-prediction thesis see “On Intelligence” by Jeff Hawkins and “How to Create a Mind“ by Ray Kurzweil. But on a technical level, they and most current researchers implement pattern discovery via artificial neural networks, which operate in a very coarse statistical fashion.
Less coarse (more selective) are Capsule Networks, recently introduced by Geoffrey Hinton et al. But they are largely ad hock, still work-in-progress, and depend on layers of CNN. Neither CNN nor CapsNet is theoretically derived. I outline my approach below, then compare it to ANN, biological NN, CapsNet and clustering, then explain my code in Implementation part.

I need help with design and implementation of this algorithm. Contributions should be justified in terms of strictly incremental search for similarity, which would form hierarchical patterns. These terms are defined below, but if you have better definitions, that would be even more valuable. This is an open project, but I will pay per contribution, or monthly if there is a good track record, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).



## Outline of my approach



Proposed algorithm is a clean design for deep learning: non-neuromorphic, sub-statistical, comparison-first. It’s a hierarchical search for patterns, by cross-comparing inputs over selectively incremental distance and composition. “Incremental” means that first-level inputs must be sub-symbolic integers with binary (before | after) coordinate.
Such as pixels of video, consecutive in each dimension, or equivalents in other modalities.

Their comparison must also be minimal in complexity, by inverse arithmetic operations. In case of subtraction, this is similar to edge detection kernel in CNN. The difference is that my comparison forms partial match along with miss and groups both into multivariate patterns: spans of same-sign miss. My match is a compression of represented magnitude by replacing inputs with misses between these inputs.

Specific match and miss (complementary of match) between two variables are determined by the power of comparison:
- Boolean match is AND and miss is XOR (two zero inputs form zero match and zero miss),
- comparison by subtraction increases match to a smaller comparand and reduces miss to a difference,
- comparison by division increases match to a multiple and reduces miss to fraction, and so on (more in part 1).

These comparisons form patterns: representations of input spans with constant sign of input-to-feedback miss.

Search hierarchy has two orders of feedback: within and between levels, forming lateral and vertical patterns. Lateral feedback is prior inputs, and their comparison forms difference patterns: spans of inputs with increasing or decreasing magnitude. Vertical feedback is average higher-level match, and comparison forms predictive value patterns: spans of inputs with above- or below- average match. Deep feedback is restricted to match: higher order of representation, to justify redundancy of value patterns to lateral difference patterns.

Higher-level inputs are patterns formed by lower-level comparisons. They represent results or derivatives: match and miss per compared input parameter. So, number of parameters per pattern is selectively multiplied on each level. Match and miss between patterns are combined matches or misses between their parameters. To maximize selectivity, search must be strictly incremental in distance, derivation, and composition over both. Which implies a unique set of operations per level of search, hence a singular in “cognitive algorithm“.

Resulting hierarchy is a dynamically extended pipeline: terminated patterns are outputted for comparison on the next level, and new level is formed for pattern terminated by current top level. Which continues as long as system receives novel inputs. As distinct from autoencoders (current mainstay in unsupervised learning), there is no need for decoding: comparison is done on each level, whose output is also fed back to filter lower levels.
Such comparison is a form of inference, and feedback of summed miss to update filters is a form of training.

To discover anything complex at “polynomial” cost, resulting patterns should also be hierarchical. In my model, each level of search adds a level of composition and a sub-level of differentiation to each input pattern.
Higher-level search is selective per level of resulting pattern. Both composition and selection speedup search, to form longer range spatio-temporal and then conceptual patterns. Which also send feedback: filters and then motor action, to select lower-level inputs and locations with above-average additive predictive value (part 3).

Hierarchical approaches are common in unsupervised learning, and all do some sort of pattern recognition.
But none that I know of is strictly incremental in scope and complexity of discoverable patterns. Which is necessary for selection, thus scalability, vs. combinatorial explosion in search space. But fine-grained selection is very expensive upfront and won’t pay off in simple test problems. So, it’s not suitable for immediate experimentation. That’s probably why no one else seems to be working on anything sufficiently similar to my algorithm.

Autonomous cognition must start with analog inputs, such as video or audio. All symbolic data, including that in natural languages, is encoded by some prior cognitive process. To discover meaningful patterns in symbols, they must be decoded before being cross-compared. And the difficulty of decoding is exponential with the level of encoding, so hierarchical learning starting with raw sensory input is by far the easiest to implement.

Such raw inputs have modality-specific properties and comparison should be adjusted accordingly. Which can be done by feedback or coded manually. For example, vision relies on reflected light: brightness or albedo don’t directly represent a source of impact, though they do represent resistance to a very "light" impact. Uniformity of albedo indicates some common underlying property, so it should form match patterns. But they should be defined in a way that doesn’t depend on magnitude of albedo. 

That means defining match through the only other derivative produced by basic comparison: as below-average absolute difference. Initial difference is across space, but it’s a product of past interactions in time (time is a macro-dimension due to a lower rate of change). Thus, difference or match across space is predictive of change or stability over time.

*Many readers feel a jarring gap between my abstract concepts and low-level code. But main principle here is selectively incremental complexity. Thus, initial code must be low, with complexity added recursively on higher levels (vs. uniform layers in ANN). Our universe is a space-time continuum, so search levels must be 4D cycles. I am currently working on initial cycle, to use as a kernel for subsequent recursion. Others note lack of higher math, but I can't use it for the same reason: it's not selectively incremental.*



### Quantifying match and miss between variables



My purpose is prediction, and predictive value is usually defined as [compressibility](https://en.wikipedia.org/wiki/Algorithmic_information_theory). Which is perfectly fine, but existing methods only compute compression per sequence of inputs. To enable more incremental selection and scalable search, I quantify partial match between individual inputs, vs. binary same | different choice for inputs within sequences. This is similar to the way Bayesian inference improved on classical logic, by quantifying probability vs. binary true | false values.

I define match as a complementary of miss. That means match is potential compression of larger comparand’s magnitude by replacing it with its miss (initially difference) relative to smaller comparand. Ultimate criterion is recorded magnitude, rather than bits of memory it occupies, because the former represents physical impact that we want to predict. The volume of memory used to record that magnitude depends on prior compression, which is not an objective parameter.

This is tautological: smaller input is a common subset of both inputs, = sum of AND between their uncompressed (unary code) representations. Some may object that match includes the case when both inputs equal zero, but then match also equals zero. Prediction is representational equivalent of physical momentum. Ultimately, we predict some potential impact on observer, represented by input. Zero input means zero impact, which has no conservable inertia, thus no intrinsic predictive value.

With incremental complexity, initial inputs have binary resolution and implicit shared coordinate (being a macro-parameter, resolution of coordinate lags that of an input). Compression of bit inputs by AND is well known as digitization: substitution of two lower 1 bits with one higher 1 bit. Resolution of coordinate (input summation span) is adjusted by feedback to form integers that are large enough to produce above-average match.

Next-order compression can be achieved by comparison between consecutive integers, distinguished by binary (before | after) coordinate. Basic comparison is inverse arithmetic operation of incremental power: AND, subtraction, division, logarithm, and so on. Additive match is achieved by comparison of a higher power than that which produced comparands: comparison by AND will not further compress integers previously digitized by AND.

Rather, initial comparison between integers is by subtraction, resulting difference is miss, and absolute match is a smaller input. For example, if inputs are 4 and 7, then miss is 3, and their match or common subset is 4. Difference is smaller than XOR (non-zero complementary of AND) because XOR may include opposite-sign (opposite-direction) bit pairs 0, 1 and 1, 0, which are cancelled-out by subtraction.

Comparison by division forms ratio, which is a magnitude-compressed difference. This compression is explicit in long division: match is accumulated over iterative subtraction of smaller comparand from remaining difference. In other words, this is also a comparison by subtraction, but between different orders of derivation. Resulting match is smaller comparand * integer part of ratio, and miss is final reminder or fractional part of ratio.

Ratio can be further compressed by converting to radix | logarithm, and so on. But computational costs may grow even faster. Thus, power of comparison should increase only for inputs sufficiently compressed by lower power: AND for bit inputs, SUB for integer inputs, DIV for pattern inputs, etc. Actual compression depends on input and on resolution of its coordinate: input | derivative summation span. We can’t control the input, so average match is adjusted via resolution of coordinate.

To filter future inputs, this absolute match should be projected: recombined with co-derived miss projected for a target distance. Filter deviation is accumulated until it exceeds the cost of updating lower-level filter. Which then forms relative match: current match - past match that co-occurs with average higher-level projected match. This relative match: above- or below- average predictive value, determines input inclusion into positive or negative predictive value pattern.

Separate filters are formed for each type of compared variable. For example, brightness of original input pixels may not be very predictive, partly because almost all perceived light is reflected rather than emitted. Then its filter will increase, reducing total span (cost) of value patterns, potentially down to 1. On the other hand, if differences or ratios between pixels are more predictive than pixels themselves, then filter for forming positive difference- or ratio- value patterns will be reduced.



## Comparison to Artificial and Biological Neural Networks



ANN learns via some version of Hebbian “fire together, wire together” coincidence reinforcement. Normally, “neuron’s” inputs are weighed at synapses, then summed and thresholded into output. Final output also back- propagates to synapses and is combined with their last input to adjust the weights. This weight adjustment is learning. But prior summation degrades resolution of inputs, precluding any comparison between them (original inputs are recoverable, but why degrade and then spend a ton of computation restoring them).

This is a statistical method: inputs are summed within samples defined by initially random weights. Weighted inputs propagate through hidden layers of ANN, then are compared to top-layer template forming an error. That error then backpropagates to train weights into meaningful values by Stochastic Gradient Descent. So, this process is comparison-last, vs. my comparison-first. Which makes it too coarse to scale without supervision or task-specific reinforcement, and it gets exponentially more coarse as ANNs become deeper. And this cycle must be repeated thousands of times because intermediate summation is horribly lossy. 

More basic problem is that SGD minimizes error (miss), which doesn’t directly correlate with maximizing match. Cognition is building a predictive model of environment, thus value of representations must be positive. Error is primarily negative, the value of reducing it can’t be directly combined with the value of increasing training scope. In CogAlg, compression increases representational capacity of a system. Nothing is randomized and resulting patterns are immediately meaningful, if tentative.

Inspiration by the brain kept ANN research going for decades before they became useful. But their “neurons” are mere stick figures for real ones. Of course, most of complexity in a neuron is due to constraints of biology. Ironically, weighted summation in ANN may also be a no-longer needed compensation for such constraint: neural memory requires dedicated connections (synapses), which makes individual comparison very expensive. But not anymore, we now have dirt-cheap random-access memory.

Other biological constraints are very slow neurons, and the imperative of fast reaction for survival in the wild. Both favor fast though crude summation (vs. slower one-to-one comparison), at the cost of glacial training. Reaction speed became less important: modern society is quite secure, while continuous learning is far more important because of accelerating progress. 
Another likely reason for the prevalence of summation in neurons is to reduce noise, because they often fire at random. That’s probably how they initially connect in the womb, and then temporarily maintain latent connections. But none of that is relevant for electronic circuits. 

In general, an algorithm should only be mapped on a network are complex enough to carry the cost of connections. Less complex representations should be stored in contiguous memory for sequential access. Cognitive function is a search for patterns, so parameters that characterize a pattern should be stored locally, while parallelizing cross-comparison among patterns is a secondary higher-order issue. 



## Comparison to Capsule Networks and Clustering



The nearest experimentally successful method is recently introduced “capsules”. Some similarities to CogAlg:
- capsules also output multi-variate vectors, “encapsulating” several properties, similar to my patterns,
- these properties also include coordinates and dimensions, compared to compute differences and ratios,
- these distances and proportions are also compared to find “equivariance” or affine transformations,
- capsules also send direct feedback to lower layer (dynamic routing), vs. trans-hidden-layer backprop in ANN

But measure of similarity in CapsNet (“agreement” in dynamic routing) is an unprincipled dot product, vs. additive compression in CogAlg. This is very common, most recognition algorithms use matrix multiplication. Conceptually, similarity is a common subset of comparands, and multiplication vastly exaggerates it by forming a superset. Exaggeration adds resistance to noise, but at the cost of drastically impaired precision. Distinction between input and noise is case-specific, it should be learned from the input itself, not built in the algorithm.    

Common subset of two integers is the smaller of them, = compression of represented magnitude by replacing larger input with the difference between inputs. This is a direct implication of information theory: compression must be a measure of similarity, but no one else seems to use it from the bottom up. It’s not sufficient per se, basic working measure would probably be more complex, but minimum is unavoidable as a starting point.
 
Some other problems I have with current implementation of CapsNet:
- CapsNet is initially fully-connected, with a network-centric bias toward uniform matrix operations, vs conditional unfolding. CogAlg search is selective over incremental dimensionality, distance, and the depth of input unfolding.
- they use CNN for initial layers, to recognize basic features, but a truly general method should apply the same principles on all levels of processing, any differentiation should be learned rather than built-in.
- capsules of all layers contain the same parameters: probability and pose variables, while I think the number of parameters should be incremental with elevation: each level forms derivatives of input parameters.
- the number of layers is fixed, while I think it should be incremental with experience.

My patterns have match instead of probability, a miss that includes pose variables, plus selected properties of lower-level patterns. In my terms, Hinton’s equivariance is a match between misses: differences and distances. 
All these variables are derived by incrementally complex comparison: core operation on all levels of CogAlg. 

My hierarchy is also dynamic: pattern is displaced from level by a miss to new input, then forwarded to existing or newly formed higher level. So, higher-level patterns include lower-level variables, as well as their derivatives. The derivatives are summed within pattern, then evaluated for extending intra-pattern search and feedback. Thus, both hierarchy of patterns per system and sub-hierarchy of variables per pattern expand with experience.

Another technique similar to mine is hierarchical clustering. But conventional clustering also defines match as inverted difference between inputs. This is the opposite of matrix multiplication, which computes match but not coincident difference. And it’s also wrong: match is a common subset of comparands, distinct from and complementary to the difference between them. Both should be computed, because each has independent predictive value.



## Implementation



Any prediction has two components: what and where. We must have both: value of prediction = precision of what * precision of where. That “where” is currently neglected: statistical ML methods represent S-T dimensions much more coarsely than the inputs. Hence, precision of where (spans of and distances between patterns) is degraded, and so is predictive value of combined representations. There is no such default degradation of positional information in my method.

Core algorithm is 1D: time only. Our space-time is 4D, and average match is presumably equal over all dimensions. That means patterns defined in fewer dimensions will be only slices of actual input, fundamentally limited and biased by the angle of scanning / slicing. Hence, initial pixel comparison should also be over 4D at once, or at least over 3D for video and 2D for still images. This full-D-cycle level of search is a universe-specific extension of core algorithm.

“Dimension” here defines external sequence and distance among inputs. This is different from conventional clustering, which treats both external and internal parameters as dimensions. 

Initial D cycle compares contiguous inputs, analogously to connected-component analysis.
Subsequent cycles will compare full-D-terminated input patterns over increasing distance in each dimension, forming discontinuous patterns of incremental composition and range. Space-time dimensions can be coded as implementation shortcut, or discovered by core algorithm itself. 

Final hierarchical algorithm will have two-level code: 
- 1st level algorithm: contiguous cross-comparison over full-D cycle, plus bit-filter feedback 
- recursive increment in complexity of comparison and feedback, generating higher-level algorithm to accommodate increasingly deep input patterns.

This repository currently has three levels of code, designed for corresponding D-cycles: 1D line algorithm, 2D frame algorithm, and 3D video algorithm. 
This code will be extended to add colors, audio, text. Initial testing could be on recognition of labeled images, but video or stereo video should be far better, more predictive representations of our 4D world.

For more detailed account of current development see [WIKI](https://github.com/boris-kz/CogAlg/wiki).

Suggestions and collaboration are most welcome, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).


