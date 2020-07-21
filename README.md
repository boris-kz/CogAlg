CogAlg
======

Full introduction: <www.cognitivealgorithm.info>

Intelligence is a general cognitive ability, ultimately the ability to predict. That includes planning, which technically is self-prediction. Any prediction is interactive projection of known patterns, which is secondary to pattern discovery. This perspective is well established, pattern recognition is a core of any IQ test. But I couldn't find a general AND constructive definition of either pattern or recognition (quantified similarity), so I came up with my own.

For excellent popular introductions to cognition-as-prediction thesis see “On Intelligence” by Jeff Hawkins and 
“How to Create a Mind“ by Ray Kurzweil. But on a technical level, they and most everyone else use ANNs, which work in very coarse statistical fashion. Capsule Networks, recently introduced by Geoffrey Hinton et al, are more local and selective by multiple instantiation parameters. But they still start with weighted summation per parameter, which degrades the data before evaluation.

In the next section, I define similarity for the simplest inputs, then describe hierarchically recursive algorithm 
of search for similarity among incrementally complex inputs: lower-level patterns. The following two sections compare my scheme to ANN, BNN, and CapsNet. 
This is an open project, we need help with design and implementation: [WIKI](https://github.com/boris-kz/CogAlg/wiki). I have awards for contributions or monthly payment if there is a track record, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md). 


### Outline of my approach


Proposed algorithm is a first-principles alternative to deep learning, neither statistical nor neuromorphic. It is designed to discover hierarchical patterns in recursively extended pipeline, with higher-composition patterns encoded on higher stages. Each stage of this pipeline has two feedforward sub-stages:
- input cross-comparison over selectively incremental distance and derivation,
- pattern composition: parameterized connectivity clustering by resulting match.

Please see [whole-system diagram](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/Whole-system%20hierarchy.png)

First-level comparands must be sensory inputs at the limit of resolution: adjacent pixels of video or equivalents in other modalities. All symbolic data is encoded by some prior cognitive process. To discover meaningful patterns in a set of symbols, they must be decoded before being cross-compared. The difficulty of decoding is exponential with the level of encoding, thus hierarchical learning that starts with raw sensory input is by far the easiest to implement (part 0).

Basic comparison is inverse arithmetic operation between single-variable comparands, of incremental power: Boolean, subtraction, division, etc. Each order of comparison forms miss or loss: XOR, difference, ratio.., and match or similarity, which can be defined directly or as inverse deviation of miss. Direct match is compression of represented magnitude by replacing larger input with corresponding order of miss between the inputs: Boolean AND, min input in comp by subtraction, integer part of ratio in comp by division, etc. (part 1). 

These direct similarity measures work if input intensity represents some conserved physical property of the source, anti-correlating with its variation. Which is the case in tactile but not in visual input: brightness doesn’t correlate with inertia or invariance, dark objects are just as stable as bright ones. So, initial match in vision should be defined indirectly, as inverse deviation of variation in intensity. 1D variation is simply difference, ratio, etc., while multi-D comparison will combine differences into Euclidean distance and gradient.

In 2D image processing, basic cross-comparison is done by edge detectors, which form gradient and its angle. They are used as first layer in the proposed model, same as in CNN. It then segments image into blobs: 2D patterns, by the sign of gradient deviation. This is also pretty conventional, but my blobs are parameterized with summed pixel-level derivatives: initially gradient and angle, and dimensions. Each parameter has independent predictive value, so they should be preserved for next-level comparison between blobs.

Higher-level inputs are lower-level patterns, their parameters are selectively cross-compared between inputs, forming match and miss per parameter. Thus, number of parameters per pattern may multiply on each level. Match and miss per pattern are summed from matches | misses per parameter, and their deviations define compositionally higher patterns. Cross-comparison is incremental in distance, derivation, and composition. Which implies a unique set of operations per level of search, hence a singular in “cognitive algorithm“.

It's a form of hierarchical connectivity clustering, defined by the results of fixed-range cross-comparison. These results must include pose parameters: coordinates and dimensions, because value of prediction = precision of what * precision of where. All parameters should be compared between patterns on a higher level, to discover longer-range spatio-temporal and then conceptual patterns. But this is too complex and slow to pay off in simple test problems, which is probably why such schemes are not actively explored.

Resulting hierarchy is a dynamic pipeline: terminated patterns are outputted for comparison on the next level, hence a new level must be formed for a pattern terminated by current top level. Which continues as long as the system receives novel inputs. As distinct from autoencoders, there is no need for decoding: comparison and clustering is done on each level. Patterns are also hierarchical: each level of search adds a level of composition and sub-level of differentiation. To avoid combinatorial explosion, search is selective per input pattern.

*Many readers see a gap between this outline and algorithm, or a lack of the latter. It’s true that the algorithm is far from complete, but above-explained principles are stable and we are translating them into code. Final algorithm will be a meta-level of search: 1st level operations plus recursive increment in input complexity, which generate next-level alg. We are in a space-time continuum, thus each level will be  3D or 4D cycle. Another complaint is that I don't use mathematical notation, but it simply doesn't have the flexibility to express deeply conditional, incrementally complex process.*



### Comparison to Artificial and Biological Neural Networks



From pattern discovery perspective, all learning is future input selection, driven by match or error from comparing processed inputs. This selection is effectively clustering inputs into patterns. In my scheme both are primarily lateral: among inputs within a level, while in statistical learning comparison and clustering are vertical: between layers of composition (weighted summation). That makes all statistical learning some form of [centroid clustering](https://en.wikipedia.org/wiki/Cluster_analysis#Centroid-based_clustering), which conceptually includes neural nets.

Basic ANN is a multi-layer perceptron: each node weighs the inputs at synapses, then sums and thresholds them into output. This normalized sum of inputs is their centroid. Output of top layer is compared to some template, forming an error. With Stochastic Gradient Descent, that error backpropagates, converting initially random weights into functional values. This is a form of learning, but I have basic problems with the process:

- Vertical learning (via feedback of error) takes tens of thousands of cycles to form accurate representations. That's because summation per layer degrades positional input resolution. Hence, the output that drives learning contains exponentially smaller fraction of original information with each added layer. My cross-comp and clustering is far more complex per level, but the output contains all information of the input. Lossy selection is only done on the next level, after evaluation per pattern (vs. before evaluation in statistical methods). 

- Both initial weights and sampling that feeds SGD are randomized. Also driven by random variation are RBMs, GANs, VAEs. Randomization is antithetical to intelligence, it's only useful in statistical methods because they merge inputs with weights irreversibly. There is no way to separate them in the output, so any non-random initialization and variation introduce artifacts and bias. All input modification in my scheme is via hyperparameters, they are stored separately and used to normalize the outputs for any difference in initial conditions. 

- SGD minimizes error (top-layer miss), which is quantitatively different from maximizing match: compression. And that error is wrt. some specific template, while my match is summed over all past input / experience. The “error” here is plural: lateral misses (differences, ratios, etc.), computed by cross-comparison within a level. All inputs represent environment and have positive value. But then they are packed (compressed) into patterns, which have different range and precision, thus different relative value per relatively fixed record cost.

- Representation in ANN is fully distributed, similar to the brain. But the brain has no alternative: there is no substrate for local memory or program in neurons. Computers have RAM, so parallelization is a simple speed vs. efficiency trade-off, useful only for complex semantically isolated nodes. Such nodes are patterns, encapsulating a set of co-derived “what” and “where” parameters. This is similar to neural ensemble, but parameters that are compared together should be localized in memory, not distributed across a network.

More basic neural learning mechanism is Hebbian: synapse is reinforced or weakened in proportion to the contribution of its input to the output. That proportion can be formalized as match of the input to normalized sum of all inputs: their centroid. Such learning is local, within each node. But it is a product of vertical comparison: centroid is a higher level of composition than individual inputs. Such comparison across composition levels drives all statistical learning. But resulting input clusters are discontinuous and overlapping, thus positional information is lost.

Inspiration by the brain kept ANN research going for decades before they became useful. Their “neurons” are mere stick figures, but that’s not a problem, most of neuron’s complexity is due to constraints of biology. The problem is that core mechanism in ANN: weighted summation, is also a no-longer needed compensation for such constraints: neural memory is dedicated connections. That makes representation and cross-comparison of individual inputs nearly impossible, so they are summed. But we now have dirt-cheap RAM.

Other biological constraints are very slow neurons, and the imperative of fast reaction for survival in the wild. Both favor fast though crude summation, at the cost of glacial training. Reaction speed became less important: modern society is quite secure, while continuous learning is far more important because of accelerating progress. Summation also reduces noise, which is very important for neurons that often fire at random, to initiate and maintain latent connections. But that’s irrelevant for electronic circuits.

Biological intelligence is a distant side effect of maximizing reproduction. The brain evolved to guide the body, with neurons originating as instinctive stimulus-to-response converters. They only do pattern discovery as instrumental upshot, not a fundamental mechanism. Hence, both SGD and Hebbian learning is fitting, driven by feedback of action, triggered by weighted input sum. 

Uri Hasson, Samuel Nastase, Ariel Goldstein made similar conclusion in “Direct fit to nature: an evolutionary perspective on biological and artificial neural networks”: “We argue that neural computation is grounded in brute-force direct fitting, which relies on over-parameterized optimization algorithms to increase predictive power (generalization) without explicitly modeling the underlying generative structure of the world. Although ANNs are indeed highly simplified models of BNNs, they belong to the same family of over-parameterized, direct-fit models, producing solutions that are mistakenly interpreted in terms of elegant design principles but in fact reflect the interdigitation of ‘‘mindless’’ optimization processes and the structure of the world.”


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


