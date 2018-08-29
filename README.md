CogAlg
======

Full introduction: www.cognitivealgorithm.info

Intelligence is the ability to predict and plan (self-predict), which can only be done by discovering and projecting patterns. This perspective is well established: pattern recognition is a core of any IQ test.
But there is no general and at the same time constructive definition of either pattern or recognition (quantified similarity) So, I came up with my own definitions, which directly translate into algorithm proposed below.

For excellent popular introductions to cognition-as-prediction thesis see “On Intelligence” by Jeff Hawkins and “How to Create a Mind“ by Ray Kurzweil. But on a technical level, they and most current researchers implement pattern discovery via artificial neural networks, which operate in a very coarse statistical fashion.
Less coarse (more selective) are Capsule Networks, recently introduced by Geoffrey Hinton et al. But they are largely ad hock, still work-in-progress, and depend on layers of CNN. Neither CNN nor CapsNet is theoretically derived. I outline my approach below, then compare it to ANN, biological NN, CapsNet and clustering, then explain my code in Implementation part.

I need help with design and implementation of this algorithm. Contributions should be justified in terms of strictly incremental search for patterns, as defined here. Unless you have a better definition, which would be even more valuable. This is an open project, but I will pay per contribution, or per hour once there is a track record, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).



## Outline of my approach



Proposed algorithm is a clean design for deep learning: non-neuromorphic, sub-statistical, comparison-first.
It’s a search for hierarchical patterns, by cross-comparing inputs over selectively incremental distance and composition. Patterns are defined by a sign of deviation of match between inputs, where match is compression of represented magnitude by replacing inputs with their derivatives. These definitions are unfolded below.

“Incremental” means that first-level comparands must be sub-symbolic integers with binary (before | after) coordinate.
Such as pixels of video, consecutive in each dimension, or equivalents in other modalities.
Their comparison should also be minimal in complexity: lossless transform by inverse arithmetic operations.
“Lossless” means that resulting match and miss are preserved as alternative representation of original inputs.

Specific match (compression of magnitude) and miss (complementary of match) are determined by the power of comparison:
- Boolean match is AND and miss is XOR (two 0 inputs form neither match nor miss),
- comparison by subtraction increases match to a smaller comparand and reduces miss to a difference,
- comparison by division increases match to a multiple and reduces miss to a fraction, and so on (more in the next section).

Resulting patterns represent spans of inputs that form same-sign miss. Hierarchy should generate two orders of feedback: within and between levels. Compared to inputs, these orders form lateral and vertical patterns:
Lateral feedback is prior inputs, and patterns are spans of inputs with increasing or decreasing magnitude.
Vertical feedback is average prior match, and patterns are spans of inputs with above- or below- average match
(deep feedback is restricted to match: higher order of representation, to justify redundancy to lateral patterns).

Higher-level inputs are patterns formed by lower-level comparisons, and they represent results or derivatives: match and miss per compared input parameter. So, the number of parameters (variables) per pattern is selectively multiplied on each level. Match and miss between patterns are combined matches or misses between their parameters. To maximize selection, search must be strictly incremental in distance, derivation, and composition over both. Which means that there is a unique set of operations per level of search, hence a singular in “cognitive algorithm“.

Resulting hierarchy is a dynamically extended pipeline: terminated patterns are outputted for comparison on the next level, and new level is formed for pattern terminated by current top level. Which continues as long as system receives novel inputs. As distinct from autoencoders (current mainstay in unsupervised learning), there is no need for decoding: comparison is done on each level, whose output is also fed back to filter lower levels.

Autonomous cognition must start with analog inputs, such as video or audio. All symbolic data, including that in natural languages, is encoded by some prior cognitive process. To discover meaningful patterns in symbols, they must be decoded before being cross-compared. And the difficulty of decoding is exponential with the level of encoding, so hierarchical learning starting with raw sensory input is by far the easiest to implement (part 0).

To discover anything complex at “polynomial” cost, resulting patterns should also be hierarchical. In my model, each level of search adds a level of composition and a sub-level of differentiation to each input pattern. Higher-level search is selective per level of resulting pattern. Both composition and selection speeds-up search, to form longer range spatio-temporal and then conceptual patterns. Which also send feedback: filters and then motor action, to select lower-level inputs and locations with above-average additive predictive value (part 3).

Hierarchical approaches are common in unsupervised learning, and all do some sort of pattern recognition.
But none that I know of is strictly incremental in scope and complexity of discoverable patterns. Which is necessary for selection, thus scalability, vs. combinatorial explosion in search space. But selection is more expensive upfront and won’t pay in simple test problems. So, it’s not suitable for immediate experimentation. That’s probably why no one else seems to be working on anything sufficiently similar to my algorithm.



### quantifying match and miss between variables



The purpose is prediction, and predictive value is usually defined as [compressibility](https://en.wikipedia.org/wiki/Algorithmic_information_theory). Which is perfectly fine, but existing methods only compute compression per sequence of inputs. To enable more incremental selection and scalable search, I quantify partial match between individual inputs, vs. binary same | different choice for inputs within sequences. This is similar to the way Bayesian inference improved on classical logic, by quantifying probability vs. binary true | false values.

I define match as a complementary of miss. That means match is potential compression of larger comparand’s magnitude by replacing it with its miss (initially difference) relative to smaller comparand. Ultimate criterion is recorded magnitude, rather than bits of memory it occupies, because the former represents physical impact that we want to predict. The volume of memory used to record that magnitude depends on prior compression, which is not an objective parameter.

This is tautological: smaller input is a common subset of both inputs, = sum of AND between their uncompressed (unary code) representations. Some may object that match includes the case when both inputs equal zero, but then match also equals zero. Prediction is representational equivalent of physical momentum. Ultimately, we predict some potential impact on observer, represented by input. Zero input means zero impact, which has no conservable inertia, thus no intrinsic predictive value.

With incremental complexity, initial inputs have binary resolution and implicit shared coordinate (being a macro-parameter, resolution of coordinate lags that of an input). Compression of bit inputs by AND is well known as digitization: substitution of two lower 1 bits with one higher 1 bit. Resolution of coordinate (input summation span) is adjusted by feedback to form integers that are large enough to produce above-average match.

Next-order compression can be achieved by comparison between consecutive integers, distinguished by binary (before | after) coordinate. Basic comparison is inverse arithmetic operation of incremental power: AND, subtraction, division, logarithm, and so on. Additive match is achieved by comparison of a higher power than that which produced comparands: comparison by AND will not further compress integers previously digitized by AND.

Rather, initial comparison between integers is by subtraction, resulting difference is miss, and absolute match is a smaller input. For example, if inputs are 4 and 7, then miss is 3, and their match or common subset is 4. Difference is smaller than XOR (non-zero complementary of AND) because XOR may include opposite-sign (opposite-direction) bit pairs 0, 1 and 1, 0, which are cancelled-out by subtraction.

Comparison by division forms ratio, which is a magnitude-compressed difference. This compression is explicit in long division: match is accumulated over iterative subtraction of smaller comparand from remaining difference. In other words, this is also a comparison by subtraction, but between different orders of derivation. Resulting match is smaller comparand * integer part of ratio, and miss is final reminder or fractional part of ratio.

Ratio can be further compressed by converting to radix | logarithm, and so on. But computational costs may grow even faster. Thus, power of comparison should increase only for inputs sufficiently compressed by lower power: AND for bit inputs, SUB for integer inputs, DIV for pattern inputs, etc. Actual compression depends on input and on resolution of its coordinate: input | derivative summation span. We can’t control the input, so average match is adjusted via resolution of coordinate.

To filter future inputs, this absolute match is projected: recombined with co-derived miss at a distance: projected match = match + (miss * distance) / 2. Filter deviation is accumulated until it exceeds the cost of updating lower-level filter. Which then forms relative match: current match - past match that co-occurs with average higher-level projected match. This relative match: above- or below- average predictive value, determines input inclusion into positive or negative pattern.

Separate filters are formed for each type of compared variable. For example, brightness of original input pixels may not be very predictive, partly because almost all perceived light is reflected rather than emitted. Then its filter will increase, reducing total span and cost of positive value patterns (vPs), potentially down to 0.
On the other hand, if differences or ratios between pixels are more predictive than pixels themselves, then the filter for forming positive difference- or ratio- value patterns (d_vPs or r_vPs) will be reduced.



## Comparison to Artificial and Biological Neural Networks



ANN learns via some version of Hebbian “fire together, wire together” coincidence reinforcement. Normally, “neuron’s” inputs are weighed at synapses, then summed and thresholded into output. Final output also back- propagates to synapses and is combined with their last input to adjust the weights. This weight adjustment is learning. But prior summation degrades resolution of inputs, precluding any comparison between them (the inputs are recoverable, but why degrade and then spend a ton of computation restoring them).

It is an inherently statistical method: inputs are summed within samples defined by initially random weights. These weights are trained into meaningful values by Stochastic Gradient Descent, but only after weighed inputs propagate through the whole network, then are compared to top-layer template to form an error, which then backpropagates through the network again. This cycle is too coarse to scale without supervision or task-specific reinforcement, especially since it must be repeated thousands of times during training.

So, ANN is a comparison-last algorithm. CogAlg is comparison-first: my initial feedback per pixel is simply a prior or adjacent pixel. This is infinitely finer-grained than backpropagation, and resulting patterns are immediately meaningful, if tentative. I also have a higher-order feedback: filters, but they are optional and there is only one filter per pattern’s variable type. ANN has specific weight for each input: a combination of template and filter. I think these are different orders of feedback, with different scope.
More broadly, SGD minimizes error (my miss), which doesn’t fully correlate with maximizing match.

Currently the most successful method is CNN, which computes similarity as a product: input * kernel (weights), adjusted by some activation function. Again, kernels start with useless random weights and their adjustment is delayed (coarsified) by hidden layers. Human brain is born with a final number of neurons. Before birth, they have to develop and connect by processing random noise, and then learn by adjusting their connections. But in software, generating and deleting nodes that represent specific content should be far more efficient.

Conceptually, similarity is a common subset of comparands, but multiplication forms a superset, which exaggerates similarity. This is compensated by some activation function, but that’s another unprincipled and grossly inaccurate hack, which causes vanishing or exploding gradients. In the brain, multiplication is analog (cheap) and exaggeration adds resistance to noise. But noise cancellation should be separate from recognition, specific to local properties of noise, which should be learned and updated along with those of input itself.

ANN compute difference  (error) on their output layer, but not between hidden layers. This distinction is only justified in supervised learning, where we have some specific expectation. In unsupervised learning, all layers are equally unknown before the input. So, each layer should compute both similarity and difference: feedback to a lower layer. In a temporal input flow, such difference should update expectations. I think it is this delayed update that causes initialization bias in ANN, and much of similar confirmation bias in humans.

Also, both input and kernel in ANN are arrays, often 2D or higher. This is far more coarse, thus less potentially selective and efficient, than one-to-one comparison of laterally adjacent inputs in my algorithm. I use “level” vs. “layer” because my levels are not identical. Complexity of inputs and operations is incremental with elevation: initial inputs are pixels, higher-level inputs are patterns formed on lower levels. My inference is comparison that forms separate match and difference, and the difference is fed back to update level-wide filters.

Inspiration by the brain kept ANN research going for decades before they became useful. But their “neurons” are mere stick figures for real ones. Of course, most of complexity in a neuron is due to constraints of biology. Ironically, weighted summation in ANN may also be a no-longer needed compensation for such constraint: neural memory requires dedicated connections (synapses), which makes individual input representation and comparison prohibitively expensive. This is no longer relevant, we now have dirt-cheap random access memory.

Other biological constraints are very slow neurons, and the imperative of fast reaction for survival in the wild. Both favor fast though crude summation (vs. slower one-to-one comparison), at the cost of glacial training. Reaction speed became less important: modern society is quite secure, while continuous learning is far more important because of accelerating progress. Another constraint is noise: neurons often fire at random, so their spikes are summed to reduce noise. Which is not a good reason to degrade far more precise electronic signals.



## Comparison to Capsule Networks and Clustering



The nearest experimentally successful method is recently introduced “capsules”. Some similarities to CogAlg:
- capsules also output multi-variate vectors, “encapsulating” several properties, similar to my patterns
- these properties also include coordinates and dimensions, compared to compute differences and ratios
- these distances and proportions are also compared to find “equivariance” or affine transformations
- capsules also send direct feedback to lower layer (dynamic routing), vs. trans-hidden-layer backprop in ANN

But measure of similarity in CapsNet (“agreement” in dynamic routing) is still an unprincipled dot product, vs. additive compression in CogAlg. This is not specific to CapsNet, most current recognition algorithms, and seemingly the brain too,  select for dot product. To repeat, multiplication vastly exaggerates similarity. Which adds noise resistance, crucial for our horribly noisy brain, but that should be a separate noise-specific function.

Pure similarity is a common subset: the smaller of comparands, = compression of represented magnitude by replacing larger comparand with its difference to a smaller comparand. This is a direct implication of information-theoretical compression-uber-alles principle, but no one else seems to use minimum as a measure of similarity. It’s not sufficient per se, basic working measure would probably be a deviation of adjusted match:
(minimum - difference/4) - average (minimum - difference/4). But minimum is unavoidable as a starting point.

Some other problems I have with current implementation of CapsNets:
- they use CNN layers to recognize basic features, but a truly general method would apply the same principles on all levels of processing, any differentiation should be learned rather than built-in.
- capsules of all layers contain the same parameters: probability and pose variables, while I think the number of parameters should be incremental with elevation.
- the number of layers is pre-determined by design, while I think it should be indefinitely incremental with experience.

My patterns have match instead of probability, a miss that includes pose variables, and selected properties of lower-level patterns. In my terms, Hinton’s equivariance is a match between misses: differences and distances. All these variables are derived by incrementally complex comparison: core operation on all levels of CogAlg.
My hierarchy is also dynamic: pattern is displaced from level by a miss to new input, then forwarded to existing or newly formed higher level. Which means that higher-level patterns include lower-level variables, as well as their derivatives. Thus, both hierarchy of patterns per system and sub-hierarchy of variables per pattern expand with experience.

Another technique similar to mine is hierarchical clustering. But conventional clustering defines match as inverted difference between inputs. This is the opposite of ANN, which computes match but not coincident difference. And it’s also wrong: match is a common subset of comparands, distinct from and complementary to the difference between them. Both should be computed, because each has independent predictive value.

*Some readers dismiss this outline as generalities, which lack a direct connection to my code. But I don’t see a disconnect, beyond simple adaptation to 2D or 3D format of the input. Please enlighten me, I will owe you big time. Of course, current code only covers first-level processing, but higher levels will be incremental in nature. Others complain about lack of math, but CogAlg must be selectively incremental, and complex math is not. And I have no fixed high-level architecture: "incremental" means that low-level code must form and reform it from the input.*



## Implementation



Any prediction has two components: what and where. We must have both: value of prediction = precision of what * precision of where. That “where” is currently neglected: statistical ML methods represent S-T dimensions with a significant lag, much more coarsely than inputs themselves. Hence, precision of where (spans of and distances between patterns) is degraded, and so is predictive value of combined representations. There is no default degradation of positional information in my method.

My core algorithm is 1D: time only. Our space-time is 4D, but each of these dimensions can be mapped on one level of search. This way, levels can select input patterns that are strong enough to justify the cost of representing additional dimension, as well as derivatives (matches and differences) in that dimension. Initial 4D cycle of search would compare contiguous inputs, analogously to connected-component analysis:

level 1 compares consecutive 0D pixels within horizontal scan line, forming 1D patterns: line segments.

level 2 compares contiguous 1D patterns between consecutive lines in a frame, forming 2D patterns: blobs.

level 3 compares contiguous 2D patterns between incremental-depth frames, forming 3D patterns: objects.

level 4 compares contiguous 3D patterns in temporal sequence, forming 4D patterns: processes.

Subsequent cycles would compare 4D input patterns over increasing distance in each dimension, forming longer-range discontinuous patterns. These cycles can be coded as implementation shortcut, or discovered by core algorithm itself, which can adapt to inputs of any dimensionality. “Dimension” here is a parameter that defines external sequence and distance among inputs. This is different from conventional clustering, which treats both external and internal parameters as dimensions.

However, average match in our space-time is presumably equal over all four dimensions. That means patterns defined in fewer dimensions will be biased by the angle of scanning, introducing artifacts. Hence, initial pixel comparison and inclusion into patterns should also be over 4D at once, or at least over 2D at once for still images. This is a universe-specific extension of my core algorithm.

Accordingly, my code here consists of three levels:

- 1D: [line_POC_introductory.py](https://github.com/boris-kz/CogAlg/blob/master/line_POC_introductory.py), which uses full variable names but is too long and dense to trace operations, and very similar but compressed and updated [line_POC.py](https://github.com/boris-kz/CogAlg/blob/master/line_POC.py), which works as intended but is not very useful in our 4D world.
- 2D: [frame_draft.py](https://github.com/boris-kz/CogAlg/blob/master/frame_draft.py), which is meant as a stand-alone 2D algorithm but is not complete, [frame_blobs.py](https://github.com/boris-kz/CogAlg/blob/master/frame_blobs.py), which will be a model for corresponding components of 3D video algorithm, and [frame_dblobs.py](https://github.com/boris-kz/CogAlg/blob/master/frame_dblobs.py), which is simplified version for debugging, currently in progress.
- 3D: [video_draft.py](https://github.com/boris-kz/CogAlg/blob/master/video_draft.py) for processing video: 2D + time. This algorithm will hopefully be effective and scalable, but is currently less than 5% done.

This algorithm will be organically extended to process color images, audio, other modalities. Symbolic data will be assigned as labels on patterns derived from analogue inputs. Initial testing could be recognition and automatic labeling of manually labeled images, but it might better to start directly with video: still images are very poor representation of our 4D world.

Suggestions and collaboration are most welcome, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).


