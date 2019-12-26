CogAlg
======

Full introduction: <www.cognitivealgorithm.info>

Intelligence is a general cognitive ability, ultimately ability to predict. That includes cognitive component of action: planning is technically a self-prediction. Any prediction is interactive projection of known patterns, hence primary cognitive process is pattern discovery. This perspective is well established, pattern recognition is a core of any IQ test. But there is no general and constructive definition of either pattern or recognition (quantified similarity). I define similarity for the simplest inputs, then design hierarchically recursive algorithm to search for similarity patterns in incrementally complex inputs: lower-level patterns.

For excellent popular introductions to cognition-as-prediction thesis see “On Intelligence” by Jeff Hawkins and “How to Create a Mind“ by Ray Kurzweil. But on a technical level, they and most current researchers implement artificial neural networks, which operate in a very coarse statistical fashion. Capsule Networks, recently introduced by Geoffrey Hinton et al, are more selective but still rely on Hebbian learning, coarse due to immediate input summation. My approach is outlined below, then compared to ANN, the brain, and CapsNet.

We need help with design and implementation of this algorithm, in Python or Julia. Current code is explained in [WIKI](https://github.com/boris-kz/CogAlg/wiki). This is an open project, but I do pay a prize for contributions, or monthly if there is a track record, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md). Contributions should be justified in terms of strictly incremental search for similarity, which forms hierarchical patterns: parameterized nearest-neighbour clusters or capsules.


### Outline of my approach


Proposed algorithm is a first-principles alternative to deep learning, non-neuromorphic and sub-statistical. It performs hierarchical search, cross-comparing inputs over selectively incremental distance and composition, followed by parameterized clustering. First-level comparands are sensory inputs at the limit of resolution: adjacent pixels of video or equivalents in other modalities. Symbolic data is second-hand, encoded and scrambled by prior processing, it should not be used as primary input in a strictly bottom-up system. 

Basic comparison is inverse arithmetic operation for two single-variable comparands, with incremental power: Boolean, subtraction, division, etc. Each order of comparison forms miss or loss: XOR, difference, ratio.., and match or similarity, which can be defined directly or as inverse deviation of miss. Direct match is compression of represented magnitude by replacing larger input with miss: AND, min input, integer part of ratio... (part 1).  Direct similarity measures work if input represents conserved property, which anti-correlates with variation. 

But it’s not the case in visual input: reflected light or albedo don’t represent physical density of observed object. Similar albedo per color does indicate some common objective property, so it should form patterns. But their predictive value doesn’t depend on intensity: dark areas can be just as  stable (invariant) as bright areas. So, initial match in vision should be defined indirectly, as inverse deviation of variation in intensity. 1D version of variation is difference, multi-D comparison will combine differences into Euclidean distance and gradient.

In 2D image processing, basic comparison is done by edge detectors, which form gradient and its angle. They are used as first layer in the proposed model, same as in CNN. It then segments image into blobs (2D patterns) by the sign of gradient deviation, which is also pretty conventional. But these blobs are parameterized with summed pixel-level intensity, derivatives (initially gradient and angle) and dimensions. Each parameter has independent predictive value, so they should be preserved for next-level comparison between blobs. I don’t know of any model that performs such parameterization, so the algorithm seems to be novel from this point on.   

Higher-level inputs are lower-level patterns, their parameters are selectively cross-compared between patterns, forming match and miss per parameter. Thus, number of parameters per pattern may multiply on each level. Match and miss per pattern are summed from matches | misses per parameter, and their deviations define compositionally higher patterns. Cross-comparison is incremental in distance, derivation, and composition. Which implies a unique set of operations per level of search, hence a singular in “cognitive algorithm“. 

It’s a form of hierarchical connectivity clustering: patterns are contiguous because they are defined by results of cross-comp, which should have a fixed range to encode pose parameters: coordinates, dimensions, orientation. This is essential because value of prediction = precision of what * precision of where. All params derived by cross-comp are predictive: cross-comp computes predictive value. They should be compared between patterns, to discover longer-range spatio-temporal and then conceptual patterns. But this process is very complex and slow, it won’t pay off in simple test problems, which is probably why such schemes are not actively explored.

Resulting hierarchy is a dynamic pipeline: terminated patterns are outputted for comparison on the next level, hence a new level must be formed for a pattern terminated by current top level. Which continues as long as the system receives novel inputs. As distinct from autoencoders, there is no need for decoding: comparison and clustering is done on each level, with threshold adjusted by feedback of summed deviation per parameter.
Patterns are also hierarchical: each level of search adds a level of composition and sub-level of differentiation to those of input pattern. To avoid combinatorial explosion, next-level search is selective per input pattern level.

Again, autonomous cognition must start with analog inputs, such as video or audio. Symbolic data: any sort of language, is encoded by some prior cognitive process. To discover meaningful patterns in a set of symbols, they must be decoded before being cross-compared. And the difficulty of decoding is exponential with the level of encoding, thus hierarchical learning starting with raw sensory input is by far the easiest to implement (part 0).

*Many readers see a gap between this outline and the algorithm, or a lack of the latter. It’s true that algorithm is far from complete, but the above-explained principles are stable, and we are translating them into code. Final algorithm will be a meta-level of search: 1st level + additional operations to process recursive input complexity increment, generating next level. We are in a space-time continuum, thus each level will be 3D or 4D cycle. I avoid complex math because it's not selectively incremental.*



### Comparison to Artificial and Biological Neural Networks



ANN learns via some version of Hebbian “fire together, wire together” coincidence reinforcement. Normally, “neuron’s” inputs are weighed at “synapses”, then summed and thresholded into input to next hidden layer.
Output of last hidden layer is compared to top-layer template, forming an error. That error backpropagates, converting initially random weights into meaningful values via Stochastic Gradient Descent. I have several basic problems with this whole paradigm, listed below along with my alternatives:

- Hebbian learning is driven by vertical input-to-output comparison, secondary to input summation. This is seductively simple per backprop cycle, but it takes tens of thousands cycles to form meaningful representations. That’s because summation is a loss of resolution, which makes learning exponentially more coarse per layer. Lateral parametrized cross-comparison is far more complex per layer, but output is immediately informative. Feedback here only adjusts layer-wide hyper-parameters: thresholds for the last step of pattern segmentation.  

- Both initial weights and sampling that feeds SGD are randomized, which is a zero-knowledge option. But we do have prior knowledge for any raw data in real space-time: proximity predicts similarity, thus search should proceed with incremental comparison range and input composition. Also driven by random variation are methods like RBM and GAN. There is nothing random in my model, that’s antithetical to intelligence. Rather, variation here is pattern projection by co-derived miss: projected input = input - (d_input * d_coordinate) / 2.

- SGD minimizes error (top-layer miss), which is quantitatively different from maximizing match: compression. And that error is w.r.t. some specific template, while my match is summed over all past input / experience. All inputs represent environment, thus have positive value. But then they are packed (compressed) into patterns, which have different range and precision, thus different representational value per relatively fixed record cost.

- Representation is fully distributed, which mimics the brain. But the brain has no alternative: no substrate for local memory or differentiated program in neurons. We have it now, parallelization in computers is a simple speed vs. efficiency trade-off, useful only for complex semantically isolated nodes. Such nodes are patterns, encapsulating a set of co-derived “what” and “where” parameters. This is similar to neural ensemble, but parameters that are compared together should be localized in memory, not distributed across a network.

Inspiration by the brain kept ANN research going for decades before they became useful. Their “neurons” are mere stick figures, but that’s not a problem, most of neuron’s complexity is due to constraints of biology. The problem is, core mechanism in ANN: weighted summation, may also be a no-longer needed compensation for such constraints. Neural memory is dedicated connections, which makes representation and cross-comparison of individual inputs very expensive, so they are summed. But we now have dirt-cheap random-access memory.

Other biological constraints are very slow neurons, and the imperative of fast reaction for survival in the wild. Both favor fast though crude summation (vs. direct parameterized clustering), at the cost of glacial training. Reaction speed became less important: modern society is quite secure, while continuous learning is far more important because of accelerating progress. Summation also reduces noise, very important for neurons that often fire at random, to initiate and maintain latent connections. But that’s irrelevant for electronic circuits.

Biological intelligence is a distant side effect of maximizing reproduction. The brain evolved to guide the body, even our abstract thinking is always framed in terms of action. Hence, Hebbian learning is driven by feedback of reaction: output of weighted input sum. Neurons evolved as instinctive stimulus-to-response converters, they only do pattern recognition as an instrumental upshot, if integrated into large networks. Primary learning is comparison-defined connectivity clustering within a level, which outputs immediately meaningful patterns.


### Comparison to Capsule Networks


The nearest experimentally successful method is recently introduced “capsules”. Some similarities to CogAlg:
- capsules also output multivariate vectors, “encapsulating” several parameters, similar to my patterns,
- these parameters also include pose: coordinates and dimensions, compared to compute corresponding miss,
- these misses / distances are compared to find affine transformations or equivariance: my match of misses,
- capsules also send direct feedback to lower layer: dynamic routing, vs. trans-hidden-layer backprop in ANN.

My main problems with CapsNet and alternative treatment:
 
- Object is defined as a recurring configuration of different parts. But such recurrence can’t be assumed, it should be derived by cross-comparing relative position among parts of matching objects. Which can only be done after their positions are cross-compared, which is after their objects are cross-compared: two levels above the level that forms initial objects. So, objects formed by positional equivariance would be secondary, though they may displace initial segmentation objects as a primary representation. Stacked Capsule Autoencoders also have exclusive segmentation on the first layer, but proximity doesn’t matter on their higher layers.

- Routing by agreement is basically recursive input clustering, by match of input vector to the output vector. The output (centroid) represents inputs at all locations, so its comparison to inputs is effectively mixed-distance. Thus, clustering in CapsNet is fuzzy and discontinuous, forming redundant representations. Routing by agreement reduces that redundancy, but not consistently so, it doesn’t specifically account for it. 
My default clustering is exclusive segmentation: each element (child) belongs to only one cluster (parent). Fuzzy clustering is selective to inputs valued above the cost of adjusting for overlap in representation, which increases with the range of cross-comparison. Conditional range increase is done on all levels of composition

- Instantiation parameters are application-specific, CapsNet has no general mechanism to derive them. My general mechanism is cross-comparison of input capsule parameters, which forms higher-order parameters. First level forms pixel-level gradient, similar to edge detection in CNN. But then it forms proximity-constrained clusters, defined by gradient and parameterized by summed pixel intensity, dy, dx, gradient, angle. This cross-comparison followed by clustering is done on all levels, with incremental number of parameters per input.

- Number of layers is fixed, while I think it should be incremental with experience. My hierarchy is a dynamic pipeline: patterns are displaced from a level by criterion sign change and sent to existing or new higher level. So, both hierarchy of patterns per system and sub-hierarchy of derivatives per pattern expand with experience. The derivatives are summed within a pattern, then evaluated for extending intra-pattern search and feedback.

- Output vector of higher capsules combines parameters of all lower layers into Euclidean distance. That is my default too, but they should also be kept separate, for potential cross-comp among layer-wide representations.
Overall, CapsNet is a variation of ANN, with input summation first and dynamic routing second. So, it’s a type of Hebbian learning, with most of the problems that I listed in the previous section.



### Quantifying match and miss between variables



The purpose here is prediction, and predictive value is usually defined as [compressibility](https://en.wikipedia.org/wiki/Algorithmic_information_theory). Which is perfectly fine, but existing methods only compute compression per sequence of inputs. To enable more incremental selection and scalable search, I quantify partial match between atomic inputs, vs. binary same | different choice for inputs within sequences. This is similar to the way Bayesian inference improved on classical logic, by quantifying probability vs. binary true | false values.

Partial match between two variables is a complementary of miss, in corresponding power of comparison: 
- Boolean match is AND and miss is XOR (two zero inputs form zero match and zero miss), 
- comparison by subtraction increases match to a smaller comparand and reduces miss to a difference,
- comparison by division increases match to min * integer part of ratio and reduces miss to a fractional part
(direct match works for tactile input. but reflected-light in vision requires inverse definition of initial match)

In other words, match is a compression of larger comparand’s magnitude by replacing it with miss. Which means that match = smaller input: a common subset of both inputs, = sum of AND between their uncompressed (unary code) representations. Ultimate criterion is recorded magnitude, rather than bits of memory it occupies, because the former represents physical impact that we want to predict. The volume of memory used to record that magnitude depends on prior compression, which is not an objective parameter. 

Given incremental complexity, initial inputs should have binary resolution and implicit shared coordinate (being a macro-parameter, resolution of coordinate lags that of an input). Compression of bit inputs by AND is well known as digitization: substitution of two lower 1 bits with one higher 1 bit. Resolution of coordinate (input summation span) is adjusted by feedback to form integers that are large enough to produce above-average match.

Next-order compression can be achieved by comparison between consecutive integers, distinguished by binary (before | after) coordinate. Basic comparison is inverse arithmetic operation of incremental power: AND, subtraction, division, logarithm, and so on. Additive match is achieved by comparison of a higher power than that which produced comparands: comparison by AND will not further compress integers previously digitized by AND.

Rather, initial comparison between integers is by subtraction, resulting difference is miss, and absolute match is a smaller input. For example, if inputs are 4 and 7, then miss is 3, and their match or common subset is 4. Difference is smaller than XOR (non-zero complementary of AND) because XOR may include opposite-sign (opposite-direction) bit pairs 0, 1 and 1, 0, which are cancelled-out by subtraction.

Comparison by division forms ratio, which is a magnitude-compressed difference. This compression is explicit in long division: match is accumulated over iterative subtraction of smaller comparand from remaining difference. In other words, this is also a comparison by subtraction, but between different orders of derivation. Resulting match is smaller comparand * integer part of ratio, and miss is final reminder or fractional part of ratio.

Ratio can be further compressed by converting to radix | logarithm, and so on. But computational costs may grow even faster. Thus, power of comparison should increase only for inputs sufficiently compressed by lower power: AND for bit inputs, SUB for integer inputs, DIV for pattern inputs, etc. Actual compression depends on input and on resolution of its coordinate: input | derivative summation span. We can’t control the input, so average match is adjusted via resolution of coordinate.

To filter future inputs, this absolute match should be projected: recombined with co-derived miss projected for a target distance. Filter deviation is accumulated until it exceeds the cost of updating lower-level filter. Which then forms relative match: current match - past match that co-occurs with average higher-level projected match. This relative match: above- or below- average predictive value, determines input inclusion into positive or negative predictive value pattern.

Separate filters are formed for each type of compared variable. Initial input, such as reflected light, is likely to be incidental and very indirectly representative of physical properties in observed objects. Then its filter will increase, reducing number of positive patterns, potentially down to 0. But differences or ratios between inputs represent variation, which is anti-correlated with match. They have negative predictive value, inverted to get incrementally closer to intrinsically predictive properties, such as mass or momentum.

Hence a vision-specific way I define initial match. Predictive visual property is albedo, which means locally stable ratio of brightness / intensity. Since lighting is usually uniform over much larger area than pixel, the difference in brightness between adjacent pixels should also be stable. Relative brightness indicates some underlying property, so it should be cross-compared to form patterns. But it is reflected: only indirectly representative of observed object.

Absent significant correlation between input magnitude and represented physical object magnitude, the only proxy to match in initial comparison is inverse deviation of absolute difference:
average_|difference| - |difference|. Though less accurate (defined via average diff vs. individual input), this match is also a complementary of diff:
- complementary of |difference| within average_|difference| (=max of the |difference| s), similar to minimum:
- complementary of |difference| within max input.



### Implementation



Any prediction has two components: what and where. We must have both: value of prediction = precision of what * precision of where. That “where” is currently neglected: statistical ML methods represent coordinates much more coarsely than the inputs. Hence, precision of where (spans of and distances between patterns) is degraded, and so is predictive value of combined representations. That's not the case here because my top-level patterns (multi-dimensional blobs) are contiguous.

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


