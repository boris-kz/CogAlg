CogAlg
======

Full introduction: www.cognitivealgorithm.info

Intelligence is a general cognitive ability, ultimately an ability to predict. That includes cognitive component of action: planning is technically self-prediction. And prediction is interactive projection of previously discovered patterns. This perspective is well established, pattern recognition is a core of any IQ test. But there is no general and constructive definition of pattern and recognition. Below, I define (quantify) similarity for the simplest inputs, then describe hierarchically recursive algorithm to search for patterns in incrementally complex inputs.

For excellent popular introductions to cognition-as-prediction thesis see “On Intelligence” by Jeff Hawkins and “How to Create a Mind“ by Ray Kurzweil. But on a technical level, they and most current researchers implement pattern discovery via artificial neural networks, which operate in a very coarse statistical fashion.
Less coarse (more selective) are Capsule Networks, recently introduced by Geoffrey Hinton et al. But they are largely ad hock, still work-in-progress, and depend on layers of CNN. Neither CNN nor CapsNet is theoretically derived. I outline my approach below, then compare it to ANN, biological NN, CapsNet and clustering. Current code is explained in  [WIKI](https://github.com/boris-kz/CogAlg/wiki).

We need help with design and implementation of this algorithm, in Python or Julia. This is an open project, but I will pay for contributions, or monthly if there is a good track record, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md). Contributions should be justified in terms of strictly incremental search for similarity, which forms hierarchical patterns. These terms are defined below, but better definitions would be an even more valuable contribution. 



## Outline of my approach



Proposed algorithm is a clean design for deep learning: non-neuromorphic, sub-statistical, comparison-first. 
It performs hierarchical search for patterns, by cross-comparing inputs over selectively incremental distance and composition. “Incremental” means that first-level inputs must be minimal in complexity, such as pixels of video or equivalents in other modalities. Symbolic data is second-hand, it should not be used as primary input. 

Pixel comparison must also be minimal in complexity: a lossless transform by inverse arithmetic operations. Initial comparison is by subtraction, similar to edge detection kernel in CNN. But my comparison forms partial match along with miss, and accumulates both inside patterns: spans of same-sign miss or match deviation. Match is compression of represented magnitude by replacing larger comparand with the miss between comparands. 

Specific match and miss between two variables are determined by the power of comparison operation: 
- Boolean match is AND and miss is XOR (two zero inputs form zero match and zero miss), 
- comparison by subtraction increases match to a smaller comparand and reduces miss to a difference,
- comparison by division increases match to a multiple and reduces miss to a fraction, and so on, see part 1.

These comparisons form patterns: representations of input spans with constant sign of input-to-feedback miss. 
Search hierarchy has two orders of feedback: within and between levels, forming lateral and vertical patterns. Lateral feedback is prior inputs, and their comparison forms difference patterns: spans of inputs with increasing or decreasing magnitude. Vertical feedback is average higher-level match, and comparison forms predictive value patterns: spans of inputs with above- or below- average match. This feedback is restricted to match: higher order of representation, to justify redundancy of value patterns to lateral difference patterns.

Higher-level inputs are patterns formed by lower-level comparisons. They represent results or derivatives: match and miss per compared input parameter. So, number of parameters per pattern is selectively multiplied on each level. Match and miss between patterns are combined matches or misses between their parameters. To maximize selectivity, search must be strictly incremental in distance, derivation, and composition over both. Which implies a unique set of operations per level of search, hence a singular in “cognitive algorithm“. 

Resulting hierarchy is a dynamic pipeline: terminated patterns are outputted for comparison on the next level, hence a new level must be formed for pattern terminated by current top level. Which continues as long as system receives novel inputs. As distinct from autoencoders (current mainstay in unsupervised learning), there is no need for decoding: comparison is done on each level, whose output is also fed back to filter lower levels. My comparison is a form of inference, and feedback of summed miss to update filters is a form of training.

To discover anything complex at “polynomial” cost, resulting patterns should also be hierarchical. In my model, each level of search adds one level of composition and one sub-level of differentiation to each input pattern. 
Higher-level search is selective per level of resulting pattern. Both composition and selection speedup search, to form longer range spatio-temporal and then conceptual patterns. Which also send feedback: filters and then motor action, to select lower-level inputs and locations with above-average additive predictive value (part 3).

Hierarchical approaches are common in unsupervised learning, and all do some sort of pattern recognition.
But none that I know of is strictly incremental in scope and complexity of discoverable patterns. Which is necessary for selectivity, thus scalability, vs. combinatorial explosion in search space. But selection is very expensive upfront and won’t pay in simple test problems. So, it’s not suitable for immediate experimentation. This is probably why no one else seems to be working on anything sufficiently similar to my algorithm.

Autonomous cognition must start with analog inputs, such as video or audio. All symbolic data (any kind of language) is encoded by some prior cognitive process. To discover meaningful patterns in symbols, they must be decoded before being cross-compared. And the difficulty of decoding is exponential with the level of encoding, thus hierarchical learning starting with raw sensory input is by far the easiest to implement (part 0). 

Such raw inputs have modality-specific properties and comparison should be adjusted accordingly, by feedback or manually. For example, vision relies on reflected light: brightness or albedo don’t directly represent source of impact, though they do represent some resistance to a very "light" impact. Uniformity of albedo indicates some common property within object, so it should form patterns. But the degree of commonality doesn’t depend on intensity of albedo, so match should be defined indirectly, as below-average |difference| or |ratio| of albedo.

*Many readers see a gap between this outline and algorithm, or a lack of the latter. It’s true that algorithm is far from complete, but the above-explained principles are stable, and we are translating them into code. Final algorithm will be a meta-level of search: 1st level + additional operations to process recursive input complexity increment, generating next level. We are in a space-time continuum, thus each level will be 3D or 4D cycle. I avoid complex math because it's not selectively incremental.*



## Comparison to Artificial and Biological Neural Networks



ANN learns via some version of Hebbian “fire together, wire together” coincidence reinforcement. Normally, “neuron’s” inputs are weighed at “synapses”, then summed and thresholded into input to next hidden layer.
Output of last hidden layer is compared to top-layer template, forming an error. That error backpropagates to train initially random weights into meaningful values by Stochastic Gradient Descent. It is a form of learning, but I see several fundamental problems with this process, hopefully resolved in my approach:

- ANN is comparison-last, vs. my comparison-first. Due to summation (loss of resolution) of weighted inputs, training process becomes exponentially more coarse with each added hidden layer. Which means that forward and feedback cycle must be repeated tens of thousands of times to achieve good results, making it too expensive to scale without supervision or task-specific reinforcement. Patterns formed by comparisons are immediately meaningful, although initially tentative, and they may send feedback from each level. But my feedback adjusts lower level-wide output filter, ~ activation function, there are no individual input filters like weights in ANN. 

- Both initial weights and sampling that feeds SGD are randomized, which is a zero-knowledge start. But we do have prior knowledge for any raw data in real space-time: proximity predicts similarity, thus search should proceed with incremental distance and input composition. Also driven by random variation are generative methods, such as RBM and GAN. But I think predictions should be generated as feedback from perceptual feedforward, there is no conceptual justification for any a priori variation, or doing anything at random.  

- SGD minimizes error, or miss to some specific template, which is bounded by zero and different from maximizing match. Error reduction is an inverse negative with diminishing returns, it can’t be combined with the value of extended training: range of correspondence. To build a predictive model of environment, value of representations must be positive: my match (compression) over all accumulated and projected input. 

- I think network-centric mindset itself is a problem. Physical network or imitation thereof is only justifiable if nodes are complex and semantically isolated enough to bear the cost of connections. Such nodes are patterns: sets of co-occurring parameter values, especially “what” and “where” parameters. They should be compared as a unit, thus encapsulated and stored in local memory. This is similar to neural ensemble, but the brain has no substrate for local memory. So, it must distribute such parameters across some network, with huge associated delays and connection costs. But for computers, parallelization is an optional speed vs. efficiency trade-off. 

Inspiration by the brain kept ANN research going for decades before they became useful. But their “neurons” are mere stick figures for real ones. Of course, most of complexity in a neuron is due to constraints of biology. Ironically, weighted summation in ANN may also be a no-longer needed compensation for such constraint:
neural memory requires dedicated connections (synapses), which makes individual input comparison very expensive. But not anymore, we now have dirt-cheap random-access memory.

Other biological constraints are very slow neurons, and the imperative of fast reaction for survival in the wild. Both favor fast though crude summation (vs. slower one-to-one comparison), at the cost of glacial training. Reaction speed became less important: modern society is quite secure, while continuous learning is far more important because of accelerating progress. Summation also reduces noise, very important because neurons often fire at random, to initiate and maintain latent connections. But that’s irrelevant for electronic circuits.



## Comparison to Capsule Networks



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
 
Search hierarchy is also dynamic: pattern is displaced from level by a miss to new input, then forwarded to existing or newly formed higher level. So, higher-level patterns include lower-level variables, as well as their derivatives. The derivatives are summed within pattern, then evaluated for extending intra-pattern search and feedback. Thus, both hierarchy of patterns per system and sub-hierarchy of variables per pattern expand with experience.



## Comparison to conventional clustering



My approach is a form of hierarchical clustering, but match in conventional clustering is inverted distance: a misleading term for difference. This is the opposite of multiplication between comparands, which computes similarity (match) but no coincident difference (miss). I believe both should be computed because each has independent predictive value: match is a common subset, distinct from and complementary to the difference.
 
This distinction is not apparent in modalities where signal carrier is “light” and reflected, such as visual input. There, magnitude (brightness) of input parameter or its match (compression of input magnitude) has low correlation with predictive value. This is true for most raw information sources, but match is a key higher-order parameter. That is, match of parameters that do represent predictive value (such as inverted distance), should be a criterion / metrics for higher-level clustering of patterns that contain / encapsulate them.

Again, main feature of my approach is incrementally deep hierarchical syntax (encapsulated parameters) of my patterns. Which means that metrics will change with elevation: criterion of higher-level clustering will be derived from comparison of lower-level parameters between their patterns. I can’t find an analogue to this evolving hierarchical nature of both elements and metrics in any other clustering technique.



## Quantifying match and miss between variables



The purpose is prediction, and predictive value is usually defined as [compressibility](https://en.wikipedia.org/wiki/Algorithmic_information_theory). Which is perfectly fine, but existing methods only compute compression per sequence of inputs. To enable more incremental selection and scalable search, I quantify partial match between atomic inputs, vs. binary same | different choice for inputs within sequences. This is similar to the way Bayesian inference improved on classical logic, by quantifying probability vs. binary true | false values.

I define match as a complementary of miss. That means match is potential compression of larger comparand’s magnitude by replacing it with its miss (initially difference) relative to smaller comparand. Ultimate criterion is recorded magnitude, rather than bits of memory it occupies, because the former represents physical impact that we want to predict. The volume of memory used to record that magnitude depends on prior compression, which is not an objective parameter.

This is tautological: smaller input is a common subset of both inputs, = sum of AND between their uncompressed (unary code) representations. Some may object that match includes the case when both inputs equal zero, but then match also equals zero. Prediction is representational equivalent of physical momentum. Ultimately, we predict some potential impact on observer, represented by input. Zero input means zero impact, which has no conservable inertia, thus no intrinsic predictive value.

With incremental complexity, initial inputs have binary resolution and implicit shared coordinate (being a macro-parameter, resolution of coordinate lags that of an input). Compression of bit inputs by AND is well known as digitization: substitution of two lower 1 bits with one higher 1 bit. Resolution of coordinate (input summation span) is adjusted by feedback to form integers that are large enough to produce above-average match.

Next-order compression can be achieved by comparison between consecutive integers, distinguished by binary (before | after) coordinate. Basic comparison is inverse arithmetic operation of incremental power: AND, subtraction, division, logarithm, and so on. Additive match is achieved by comparison of a higher power than that which produced comparands: comparison by AND will not further compress integers previously digitized by AND.

Rather, initial comparison between integers is by subtraction, resulting difference is miss, and absolute match is a smaller input. For example, if inputs are 4 and 7, then miss is 3, and their match or common subset is 4. Difference is smaller than XOR (non-zero complementary of AND) because XOR may include opposite-sign (opposite-direction) bit pairs 0, 1 and 1, 0, which are cancelled-out by subtraction.

Comparison by division forms ratio, which is a magnitude-compressed difference. This compression is explicit in long division: match is accumulated over iterative subtraction of smaller comparand from remaining difference. In other words, this is also a comparison by subtraction, but between different orders of derivation. Resulting match is smaller comparand * integer part of ratio, and miss is final reminder or fractional part of ratio.

Ratio can be further compressed by converting to radix | logarithm, and so on. But computational costs may grow even faster. Thus, power of comparison should increase only for inputs sufficiently compressed by lower power: AND for bit inputs, SUB for integer inputs, DIV for pattern inputs, etc. Actual compression depends on input and on resolution of its coordinate: input | derivative summation span. We can’t control the input, so average match is adjusted via resolution of coordinate.

To filter future inputs, this absolute match should be projected: recombined with co-derived miss projected for a target distance. Filter deviation is accumulated until it exceeds the cost of updating lower-level filter. Which then forms relative match: current match - past match that co-occurs with average higher-level projected match. This relative match: above- or below- average predictive value, determines input inclusion into positive or negative predictive value pattern.

Separate filters are formed for each type of compared variable. For example, brightness of original input pixels may not be very predictive, partly because almost all perceived light is reflected rather than emitted. Then its filter will increase, reducing total span (cost) of value patterns, potentially down to 1. On the other hand, if differences or ratios between pixels are more predictive than pixels themselves, then filter for forming positive difference- or ratio- value patterns will be reduced.



## Implementation



Any prediction has two components: what and where. We must have both: value of prediction = precision of what * precision of where. That “where” is currently neglected: statistical ML methods represent coordinates much more coarsely than the inputs. Hence, precision of where (spans of and distances between patterns) is degraded, and so is predictive value of combined representations. That's not the case here because my top-level patterns are contiguous.

Core algorithm is 1D: time only. Our space-time is 4D, and average match is presumably equal over all dimensions. That means patterns defined in fewer dimensions will be only slices of actual input, fundamentally limited and biased by the angle of scanning / slicing. Hence, initial pixel comparison should also be over 4D at once, or at least over 3D for video and 2D for still images. This full-D-cycle level of search is a universe-specific extension of core algorithm. The dimensions should be discoverable by the core algorithm, but coding it in is much faster. 

This repository currently has three versions of 1st D-cycle, analogous to connected-component analysis: 1D line alg, 2D frame alg, and 3D video alg.
Subsequent cycles will compare full-D-terminated input patterns over increasing distance in each dimension, forming discontinuous patterns of incremental composition and range.
“Dimension” here defines external sequence and distance among inputs. This is different from conventional clustering, which treats both external and internal parameters as dimensions. 

Complete hierarchical algorithm will have two-level code: 
- 1st level algorithm: contiguous cross-comparison over full-D cycle, plus feedback to adjust most and least significant bits of the input. 
- Recurrent increment in complexity, extending current-level alg to next-level alg. This increment will account for increasing internal complexity of input patterns on higher levels, unfolding them for cross-comparison and re-folding results for evaluation and feedback.

Initial testing could be on recognition of labeled images, but video or stereo video should be much better. We will then add colors, maybe audio and text. 

For more detailed account of current development see [WIKI](https://github.com/boris-kz/CogAlg/wiki).

Suggestions and collaboration are most welcome, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).


