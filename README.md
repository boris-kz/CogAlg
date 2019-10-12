CogAlg
======

Full introduction: www.cognitivealgorithm.info

Intelligence is a general cognitive ability, ultimately an ability to predict. That includes cognitive component of action: planning is technically a self-prediction. Any prediction is interactive projection of known patterns, hence primary cognitive process is pattern discovery. This perspective is well established, pattern recognition is a core of any IQ test. But there is no general and constructive definition of either pattern or recognition (quantified similarity). Below, I define similarity for the simplest inputs, then describe hierarchically recursive algorithm to search for patterns across incrementally complex inputs (lower-level patterns).

For excellent popular introductions to cognition-as-prediction thesis see “On Intelligence” by Jeff Hawkins and “How to Create a Mind“ by Ray Kurzweil. But on a technical level, they and most current researchers implement pattern discovery via artificial neural networks, which operate in a very coarse statistical fashion.
Less coarse (more selective) are Capsule Networks, recently introduced by Geoffrey Hinton et al. But they are largely ad hock and still work-in-progress. My approach is derived from theoretically defined measure of similarity, I outline it below and then compare to ANN, biological NN and CapsNet. Current code is explained in [WIKI](https://github.com/boris-kz/CogAlg/wiki).

We need help with design and implementation of this algorithm, in Python or Julia. This is an open project, but I will pay for contributions, or monthly if there is a good track record, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md). Contributions should be justified in terms of strictly incremental search for similarity, which forms hierarchical patterns. These terms are defined below, but better definitions would be even more valuable contribution. 



### Outline of my approach



Proposed algorithm is a clean-design alternative to deep learning: non-neuromorphic, sub-statistical, comparison-first. It performs hierarchical search for patterns, by cross-comparing inputs over selectively incremental distance and composition. Hence, first-level comparands must be minimal in complexity and distance from each other, such as adjacent pixels of video or equivalents in other modalities. Symbolic data is second-hand, it should not be used as primary input for any self-contained system. 

Pixel comparison must also be minimal in complexity: a lossless transform by inverse arithmetic operations. Initial comparison is by subtraction, similar to edge detection kernel in CNN. But my cross-comparison forms partial match and angle along with gradient: each of these derivatives has independent predictive value. Derivatives are summed in patterns: representations of input spans with same-sign miss or match deviation. Higher-level cross-comparison (~convolution) is among these multi-parameter patterns, not initial inputs.

Match is a compression of represented magnitude by replacing larger input with the miss between inputs. Specific match and miss between two variables are determined by the power of comparison operation: 
- Boolean match is AND and miss is XOR (two zero inputs form zero match and zero miss), 
- comparison by subtraction increases match to a smaller comparand and reduces miss to a difference,
- comparison by division increases match to a multiple and reduces miss to a fraction, and so on, see part 1.

Deviations that define patterns are relative to filters: initially average corresponding derivative, then updated by feedback of summed deviations. Higher-level inputs are lower-level patterns, and new patterns include match and miss per selected parameter of an input pattern. So, number of parameters per pattern is selectively multiplied on each level, and match and miss per pattern is summed match or miss per constituent parameter. To maximize selectivity, search must be strictly incremental in distance, derivation, and composition over both. Which implies a unique set of operations per level of search, hence a singular in “cognitive algorithm“.

Resulting hierarchy is a dynamic pipeline: terminated patterns are outputted for comparison on the next level, hence a new level must be formed for pattern terminated by current top level. Which continues as long as system receives novel inputs. As distinct from autoencoders (current mainstay in unsupervised learning), there is no need for decoding: comparison is done on each level, whose output is also fed back to filter lower levels. My comparison is a form of inference, and feedback of summed miss to update filters is a form of training.

To discover anything complex at “polynomial” cost, resulting patterns should also be hierarchical. In my model, each level of search adds one level of composition and one sub-level of differentiation to each input pattern. 
Higher-level search is selective per level of resulting pattern. Both composition and selection speedup search, to form longer range spatio-temporal and then conceptual patterns. Which also send feedback: filters and then motor action, to select lower-level inputs and locations with above-average additive predictive value (part 3).

Hierarchical approaches are common in unsupervised learning, and all do some sort of pattern recognition.
But none that I know of is strictly incremental in scope and complexity of discoverable patterns. Which is necessary for selectivity, thus scalability, vs. combinatorial explosion in search space. But selection is very expensive upfront and won’t pay in simple test problems. So, it’s not suitable for immediate experimentation. This is probably why no one else seems to be working on anything sufficiently similar to my algorithm.

Autonomous cognition must start with analog inputs, such as video or audio. All symbolic data (any kind of language) is encoded by some prior cognitive process. To discover meaningful patterns in symbols, they must be decoded before being cross-compared. And the difficulty of decoding is exponential with the level of encoding, thus hierarchical learning starting with raw sensory input is by far the easiest to implement (part 0). 

Such raw inputs have modality-specific properties and comparison should be adjusted accordingly, by feedback or manually. For example, vision relies on reflected light: brightness or albedo don’t directly represent source of impact, though they do represent some resistance to a very "light" impact. Uniformity of albedo indicates some common property within object, so it should form patterns. But the degree of commonality doesn’t depend on intensity of albedo, so match should be defined indirectly, as below-average |difference| or |ratio| of albedo.

*Many readers see a gap between this outline and the algorithm, or a lack of the latter. It’s true that algorithm is far from complete, but the above-explained principles are stable, and we are translating them into code. Final algorithm will be a meta-level of search: 1st level + additional operations to process recursive input complexity increment, generating next level. We are in a space-time continuum, thus each level will be 3D or 4D cycle. I avoid complex math because it's not selectively incremental.*



### Comparison to Artificial and Biological Neural Networks



ANN learns via some version of Hebbian “fire together, wire together” coincidence reinforcement. Normally, “neuron’s” inputs are weighed at “synapses”, then summed and thresholded into input to next hidden layer.
Output of last hidden layer is compared to top-layer template, forming an error. That error backpropagates to train initially random weights into meaningful values by Stochastic Gradient Descent. It is a form of learning, but I see several fundamental problems with this process, hopefully resolved in my approach:

- ANN is comparison-last, vs. my comparison-first. Due to summation (loss of resolution) of weighted inputs, training process becomes exponentially more coarse with each added hidden layer. Which means that forward and feedback cycle must be repeated tens of thousands of times to achieve good results, making it too expensive to scale without supervision or task-specific reinforcement. Patterns formed by comparisons are immediately meaningful, although tentative and refined by filter feedback. There is no need for individual weights per node.

- Both initial weights and sampling that feeds SGD are randomized, which is a zero-knowledge option. But we do have prior knowledge for any raw data in real space-time: proximity predicts similarity, thus search should proceed with incremental distance and input composition. Also driven by random variation are generative methods, such as RBM and GAN. But I think predictions should be a feedback from perceptual feedforward, there is no conceptual justification for any a priori variation, or doing anything at random.  

- SGD minimizes error (miss), producing a double negative bounded by zero. This is quantitatively different from maximizing match, defined here as a minimum. Also, that error is to some specific template, while my match is defined over all accumulated and projected input. All input representations have positive value, they combine into prediction of known “universe”. This combination is compressive for matches between similar projections, and negating for interferance between conflicting projections, both per target location.

- Presumption of some fixed nodes is confusing hardware with algorithm. There is no such distinction in the brain: neurons are both, with no substrate for local memory or differentiated algorithm. Biology has to use cells: generic autonomous nodes, because it evolved by replication. So, it distributes representations across a network of such nodes, with huge associated delays and connection costs. But parallelization in computers is a simple speed vs. efficiency trade-off, useful only for complex semantically isolated “data nodes”, AKA patterns.

Patterns contain a set of co-derived parameters, combining content’ “what” and dimensions’ “where”. This is functionally similar to neural ensemble, but parameters that are compared together should be in local memory, vs. distributed across a network. Patterns’ syntax and composition is incremental, starting with a grid of pixels and adding higher levels while learning. Compare that to brain, born with a relatively fixed number of nodes and layers. Even aside from its crude training process, such network is initially excessive and ultimately limiting.

Inspiration by the brain kept ANN research going for decades before they became useful. But their “neurons” are mere stick figures for real ones. Of course, most of complexity in a neuron is due to constraints of biology. Ironically, weighted summation in ANN may also be a no-longer needed compensation for such constraint:
neural memory requires dedicated connections (synapses), which makes individual input comparison very expensive. But not anymore, we now have dirt-cheap random-access memory.

Other biological constraints are very slow neurons, and the imperative of fast reaction for survival in the wild. Both favor fast though crude summation (vs. slower one-to-one comparison), at the cost of glacial training. Reaction speed became less important: modern society is quite secure, while continuous learning is far more important because of accelerating progress. Summation also reduces noise, very important because neurons often fire at random, to initiate and maintain latent connections. But that’s irrelevant for electronic circuits.



### Comparison to Capsule Networks



The nearest experimentally successful method is recently introduced “capsules”. Some similarities to CogAlg:
- Capsules also output multi-variate vectors, “encapsulating” several parameters, similar to my patterns,
- These parameters also include coordinates and dimensions, compared to compute differences and ratios (proportions and orientation),
- These distances and proportions are also compared to find “equivariance” or affine transformations,
- Capsules also send feedback to lower layer (dynamic routing), vs. trans-hidden-layer backprop in ANN

My main problems with CapsNet and alternative treatment in CogAlg:
- Parameters are not consistently derived by incremental and recursive cross-comparisons, starting with pixels.
- Capsules of all layers have the same parameters, while I think their number should be incremental with elevation, each level adding new derivatives per input parameter.
- Main parameters are probability and pose variables. My patterns have match instead of probability and miss that includes spatial distance, which is converted into pose variables by cross-comparison. Equivariance is my match among misses: differences and distances.

- Number of layers is fixed, while I think it should be incremental with experience. Hierarchy should be a dynamic pipeline: pattern is displaced from a current level by a miss to new input, then forwarded to existing or newly formed higher level. Thus, both hierarchy of patterns per system, and sub-hierarchy of variables per pattern, will expand with experience. The derivatives are summed within pattern, then evaluated for extending intra-pattern search and feedback.

- Measure of similarity in CapsNet, and “agreement” in dynamic routing, is still an unprincipled dot product. Product vastly exaggerates similarity. It's a superset of comparands magnitude, but similarity is conceptually a common subset, which would be a minimum for single-variable comparands. Exaggeration adds resistance to noise, but at the cost of drastically impaired precision. The distinction between signal and noise is case-specific and should be learned from the input, not built into algorithm.

Dot product is a currently dominant similarity measure, but it has no theoretical justification. I think one of the reasons this exaggerated similarity is so subjectively effective is a winner-take-all bias of most recognition tests: a single central object on a background. This might be related to “singular” focus in biological perception: it evolved to guide a single body in the environment. Which is amplified by dominant focus on interpersonal and hunter-prey interactions in humans: these are mostly one-to-one. But it’s still wrong as general principle.

My approach is a form of nearest-neighbour hierarchical clustering, but main feature is incrementally deep syntax (encapsulated parameters) per pattern. Which means that clustering metrics will change with elevation: criterion of higher-level clustering will be derived from comparison of lower-level parameters. I can’t find any other clustering technique with such evolving hierarchical nature of both elements and metrics.



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


