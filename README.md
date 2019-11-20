CogAlg
======

Full introduction: <www.cognitivealgorithm.info>

Intelligence is a general cognitive ability, ultimately ability to predict. That includes cognitive component of action: planning is technically a self-prediction. Any prediction is interactive projection of known patterns, hence primary cognitive process is pattern discovery. This perspective is well established, pattern recognition is a core of any IQ test. But there is no general and constructive definition of either pattern or recognition (quantified similarity). I define similarity for the simplest inputs, then design hierarchically recursive algorithm to search for similarity patterns in incrementally complex inputs: lower-level patterns.

For excellent popular introductions to cognition-as-prediction thesis see “On Intelligence” by Jeff Hawkins and “How to Create a Mind“ by Ray Kurzweil. But on a technical level, they and most current researchers implement artificial neural networks, which operate in a very coarse statistical fashion. Capsule Networks, recently introduced by Geoffrey Hinton et al, are more selective but still rely on Hebbian learning, with coarse immediate input summation. My approach is outlined below, then compared to ANN, the brain, and CapsNet.

We need help with design and implementation of this algorithm, in Python or Julia. Current code is explained in [WIKI](https://github.com/boris-kz/CogAlg/wiki). This is an open project, but I do pay a prize for contributions, or monthly if there is a track record, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md). Contributions should be justified in terms of strictly incremental search for similarity, which forms hierarchical patterns: parameterized nearest-neighbour clusters or capsules.



### Outline of my approach



Proposed algorithm is a clean-design alternative to deep learning: non-neuromorphic and sub-statistical. It performs hierarchical search, cross-comparing inputs over selectively incremental distance and composition, followed by parameterized clustering. First-level comparands are sensory inputs at the limit of resolution: adjacent pixels of video or equivalents in other modalities. Symbolic data is second-hand, encoded and scrambled by prior processing, thus it should not be used as primary input in a strictly bottom-up system. 

Initial cross-comparison must be minimal in complexity: a lossless transform by inverse arithmetic operations. Basic comparison by subtraction forms gradient, similar to edge detection kernel in CNN. But my model also forms partial match, angle, and dimensions: each of them has independent predictive value. These derivatives are summed in patterns: representations of input areas with same-sign miss or match deviation. Higher-level cross-comparison will be among multi-parameter patterns formed on a lower level, not among initial inputs.

Match is compression of represented magnitude by replacing larger input with the miss (loss or error) between inputs. Specific match and miss between two variables are determined by the power of comparison: 
- Boolean match is AND and miss is XOR (two zero inputs form zero match and zero miss), 
- comparison by subtraction increases match to a smaller comparand and reduces miss to a difference,
- comparison by division increases match to min * integer part of ratio and reduces miss to a fractional part
And so on, see part 1. These are 1D versions,  subtraction in real space will form Euclidean distance, etc.

These direct similarity measures are appropriate for conserved parameters, which anti-correlate with variation. But that can’t be assumed in initial inputs: reflected light or albedo don’t represent physical density of observed object. Similar albedo per color does indicate some common objective property, so it should form patterns. But their predictive value doesn’t depend on intensity: dark areas can be just as  stable (invariant) as bright areas. Hence, initial match should be defined indirectly, as inverse deviation of variation in the input.

Patterns are defined by deviations from corresponding filters: initially average derivative, updated by feedback of deviation summed on a higher level. Higher-level inputs are lower-level patterns, and new patterns include match and miss per selected parameter of an input pattern. So, number of parameters per pattern is selectively multiplied on each level, and match and miss per pattern is summed match or miss per constituent parameter. To maximize selectivity, search must be strictly incremental in distance, derivation, and composition over both. Which implies a unique set of operations per level of search, hence a singular in “cognitive algorithm“. 

Resulting hierarchy is a dynamic pipeline: terminated patterns are outputted for comparison on the next level, hence a new level must be formed for a pattern terminated by current top level. Which continues as long as the system receives novel inputs. As distinct from autoencoders, there is no need for decoding: comparison and clustering is done on each level, with the threshold adjusted by feedback of summed deviation per parameter.
Patterns are also hierarchical: each level of search adds a level of composition and sub-level of differentiation to each input pattern. Next-level search is selective per each level of a pattern, to avoid combinatorial explosion.

This is a form of hierarchical nearest-neighbour clustering, which is another term for pattern discovery. But I don’t know of any approach that cross-compares multiple parameters per element, forming new derivatives that are summed into cluster parameters. These parameters are necessary for accurate selection and scalability to long-range spatio-temporal and then conceptual patterns. But processing is very complex and slow upfront. It won’t pay off for simple test problems, which is probably why such schemes are not actively explored.

Again, autonomous cognition must start with analog inputs, such as video or audio. Symbolic data: any sort of language, is encoded by some prior cognitive process. To discover meaningful patterns in a set of symbols, they must be decoded before being cross-compared. And the difficulty of decoding is exponential with the level of encoding, thus hierarchical learning starting with raw sensory input is by far the easiest to implement (part 0). 

*Many readers see a gap between this outline and the algorithm, or a lack of the latter. It’s true that algorithm is far from complete, but the above-explained principles are stable, and we are translating them into code. Final algorithm will be a meta-level of search: 1st level + additional operations to process recursive input complexity increment, generating next level. We are in a space-time continuum, thus each level will be 3D or 4D cycle. I avoid complex math because it's not selectively incremental.*



### Comparison to Artificial and Biological Neural Networks



ANN learns via some version of Hebbian “fire together, wire together” coincidence reinforcement. Normally, “neuron’s” inputs are weighed at “synapses”, then summed and thresholded into input to next hidden layer.
Output of last hidden layer is compared to top-layer template, forming an error. That error backpropagates, converting initially random weights into meaningful values via Stochastic Gradient Descent. 
I have several basic problems with this whole neural paradigm, listed below along with my alternatives:

- Hebbian learning is reactive: neurons evolved primarily for stimulus-to-response conversion, via weighed input summation. Learning here is secondary, driven by input-to-output comparison. Summation is a loss of resolution, which makes training exponentially more coarse with each added layer. This design is seductively simple and fast per feedforward / backprop cycle, but it takes tens of thousands of cycles to form meaningful representations. 
Parametrized cross-comparison is far more complex per cycle, but is immediately informative. Feedback here only adjusts filters that define patterns, for selective recursively deeper cross-comparison.  

- Both initial weights and sampling that feeds SGD are randomized, which is a zero-knowledge option. But we do have prior knowledge for raw data in real space-time: proximity predicts similarity, thus search should proceed with incremental distance and input composition. Also driven by random variation are methods like RBM and GAN. There is a place for variation in my model, but it’s never random. Rather, variation is pattern projection by co-derived miss: projected input = input - (d_input * d_coordinate) / 2.

- SGD minimizes error (top-layer miss), which is quantitatively different from maximizing match: compression. And that error is wrt. some specific template, while my match is summed over all accumulated input. All inputs represent environment, thus have positive value. But then they are packed (compressed) into patterns, which have different range and precision, thus different representational value per relatively fixed record cost.

- Representation is fully distributed, which mimics the brain. But the brain has no alternative: no substrate for local memory or differentiated program in neurons. We have it now, parallelization in computers is a simple speed vs. efficiency trade-off, useful only for complex semantically isolated nodes. Such nodes are patterns, encapsulating a set of co-derived “what” and “where” parameters. This is similar to neural ensemble, but parameters that are compared together should be localized in memory, not distributed across a network.

Inspiration by the brain kept ANN research going for decades before they became useful. Their “neurons” are mere stick figures, but that’s not a problem, most of neuron’s complexity is due to constraints of biology. The problem is, core mechanism in ANN: weighted summation, may also be a no-longer needed compensation for such constraints. Neural memory is dedicated connections, which makes representation and cross-comparison of individual inputs very expensive, so they are summed. But we now have dirt-cheap random-access memory.

Other biological constraints are very slow neurons, and the imperative of fast reaction for survival in the wild. Both favor fast though crude summation (vs. direct parameterized clustering), at the cost of glacial training. Reaction speed became less important: modern society is quite secure, while continuous learning is far more important because of accelerating progress. Summation also reduces noise, very important for neurons that often fire at random, to initiate and maintain latent connections. But that’s irrelevant for electronic circuits.

Biological intelligence is merely a distant side effect of evolution. The brain evolved to guide the body and our feeble abstract thinking is always framed in terms of action. Neurons themselves started as primarily instinctive stimulus-response systems, with complex pattern recognition as an instrumental upshot. Hence, Hebbian learning is secondary to reaction: driven by feedback of summed output. In my scheme, learning is feedforward comparison-defined clustering, acting on input-level resolution and primary to any output.



### Comparison to Capsule Networks



The nearest experimentally successful method is recently introduced “capsules”. Some similarities to CogAlg:
- capsules also output multi-variate vectors, “encapsulating” several parameters, similar to my patterns,
- these parameters also include pose: coordinates and dimensions, compared to compute corresponding miss,
- these misses / distances are compared to find affine transformations or equivariance: my match of misses,
- capsules also send direct feedback to lower layer: dynamic routing, vs. trans-hidden-layer backprop in ANN.

My main problems with CapsNet and alternative treatment:
 
- Instantiation parameters are defined depending on application, there is no general mechanism for that. 
My parameters are derived by cross-comparison of input capsule parameters. First level derives pixel-level gradient, similar to edge detection in CNN. But then it forms clusters: constrained by proximity, defined by gradient, and parameterized by summed pixel intensity, dy, dx, gradient, and angle (all per color). This cross-comparison followed by clustering is done on all levels, with incremental number of parameters per input.

- Object is defined as a recurring configuration of different parts. But such recurrence can’t be assumed, it should be derived by cross-comparing relative position among parts of matching objects. Which can only be done after their positions are cross-compared, which is after their objects are cross-compared: two levels above the level that forms initial objects. So, objects formed by positional equivariance would be secondary. But they may be stronger, displacing initial similarity-defined objects as a primary representation of the same parts.   

- This re-clustering by positional equivariance in my model replaces routing by agreement: re-clustering by match of input vector to output vector in CapsNet. Such “vertical” match is a product of effectively mixed-distance comparison: output (centroid) represents inputs at different locations. So, search is not exactly proximity-ordered, doesn’t form one-to-one distances between nearest parts, and resulting clusters won’t be contiguous. I think this is contrary to main principle of CapsNet: preserving positional information.

- “Part” clustering in CapsNet is fuzzy and discontinuous, forming redundant representations. Routing by agreement reduces that redundancy, but not consistently: it doesn’t specifically account for it. 
My default clustering is exclusive segmentation: each element (child) belongs to only one cluster (parent). Fuzzy clustering is selective to inputs with value above the cost of adjusting for overlap in representation, increasing with the range of cross-comparison. Such range incrementing is done on all levels of composition.
(Stacked Capsule Autoencoders do have exclusive segmentation on the first layer, but not on higher layers)

- Pose parameters are derived by CNN layers. The first convo-layer performs cross-correlation with edge detection kernel, similar to my model, but it’s not followed by parametrised clustering. Higher convo-layers only increase kernel area, while my higher lateral cross-comparison is among parameters of lower-layer clusters, followed by higher clustering. There may be vertical comparison to short-cuts across the hierarchy, to speed-up comparison that would happen after sequential propagation of input through intermediate levels.

- Number of layers is fixed, while I think it should be incremental with experience. My hierarchy is a dynamic pipeline: patterns are displaced from a level by criterion sign change and sent to existing or new higher level. So, both hierarchy of patterns per system and sub-hierarchy of derivatives per pattern expand with experience. The derivatives are summed within pattern, then evaluated for extending intra-pattern search and feedback.

- Output vector of higher capsules combines parameters of all lower layers into Euclidean distance. That is my default too, but they should also be kept separate, for potential cross-comp among layer-wide representations within each capsule. This would be similar to frequency domain representation, but specific to each pattern. 

So, all versions of CapsNet perform input summation first and Hebbian-type learning second. That learning is basically vertical comparison of vectors’ “squashed” length, between each input and combined output. The length indicates probability (actually similarity) of some predefined capsule. This vertical comparison drives dynamic routing, recursively re-clustering the inputs. This is far more coarse than my lateral comparison first.
Overall, CapsNet is a variation of ANN, with most of the problems that I listed in the previous section.




### Quantifying match and miss between variables



The purpose is prediction, and predictive value is usually defined as [compressibility](https://en.wikipedia.org/wiki/Algorithmic_information_theory). Which is perfectly fine, but existing methods only compute compression per sequence of inputs. To enable more incremental selection and scalable search, I quantify partial match between atomic inputs, vs. binary same | different choice for inputs within sequences. This is similar to the way Bayesian inference improved on classical logic, by quantifying probability vs. binary true | false values.

I define match as a complementary of miss. That means match is potential compression of larger comparand’s magnitude by replacing it with its miss (initially difference) relative to smaller comparand. Ultimate criterion is recorded magnitude, rather than bits of memory it occupies, because the former represents physical impact that we want to predict. The volume of memory used to record that magnitude depends on prior compression, which is not an objective parameter.

This is tautological: smaller input is a common subset of both inputs, = sum of AND between their uncompressed (unary code) representations. Some may object that match includes the case when both inputs equal zero, but then match also equals zero. Prediction is representational equivalent of some physical momentum. Ultimately, we predict potential impact on observer, represented by input. Zero input means zero impact, which has no conservable inertia, thus no intrinsic predictive value.

With incremental complexity, initial inputs have binary resolution and implicit shared coordinate (being a macro-parameter, resolution of coordinate lags that of an input). Compression of bit inputs by AND is well known as digitization: substitution of two lower 1 bits with one higher 1 bit. Resolution of coordinate (input summation span) is adjusted by feedback to form integers that are large enough to produce above-average match.

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


