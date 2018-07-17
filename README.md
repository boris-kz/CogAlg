CogAlg
======

Full introduction: www.cognitivealgorithm.info

Intelligence is the ability to predict and plan, which can only be shown by discovering and projecting patterns. This definition is well established: pattern recognition is core for any IQ test. But there is no general constructive definition of neither pattern nor recognition. So I came up with my own definitions, which directly translates into the algorithm introduced here.

For excellent and popular introductions to the cognition-as-prediction thesis see “On Intelligence” by Jeff Hawkins and “How to Create a Mind“ by Ray Kurzweil. But on a technical level, they operate in a very statiscally coarse fashion because most current researchers implement pattern discovery via artificial neural networks.
Capsule Networks, recently introduced by Geoffrey Hinton et al, are less coarse and more selective in comparison. But they are largely ad hock, still work-in-progress, and depend on layers of CNN. Neither CNN nor CapsNet is theoretically derived. I outline my approach below and then compare it to ANN, biological NN, CapsNet, and clustering.

I need help with design and implementation of this algorithm, in Python. But this work is theory first. If you find flaws or omissions in my reasoning, that would be immensely valuable. This is an open project, but I will pay for contributions, or monthly if there are some track record. Please contact me if interested, here or on G+.



## Outline of my approach



Proposed algorithm is a clean design for deep learning: non-neuromorphic, sub-statistical, comparison-first. It’s a search for hierarchical patterns, by cross-comparing inputs over selectively incremental distance and composition. Patterns are defined by a sign of deviation of match between inputs, where match is compression of represented magnitude by replacing inputs with their derivatives. These definitions are unfolded below.

“Incremental” means that first-level comparands must be sub-symbolic integers with binary (before | after) coordinate. Such as pixels of video, consecutive in each dimension, or equivalents in other modalities.
Their comparison must also be minimal in complexity: lossless transform by inverse arithmetic operations. 
“Lossless” means that resulting match and miss are preserved as alternative representation of original inputs.

Specific match and miss are determined by the power of comparison: Boolean match is AND and miss is XOR, comparison by subtraction increases match to a smaller comparand and reduces miss to a difference, comparison by division increases match to multiple and reduces miss to fraction, and so on (more in part 1). Generalizing the above, match is lossless compression per comparison, /= redundancy in input representation.

Resulting patterns represent spans of inputs that form same-sign miss. Hierarchy should generate two orders of feedback: within and between levels. Compared to inputs, these orders form lateral and vertical patterns: Lateral feedback is prior inputs, and patterns are spans of inputs with increasing or decreasing magnitude. Vertical feedback is average prior match, and patterns are spans of inputs with above- or below- average match (this feedback is restricted to match: higher order of representation, to justify redundancy to lateral patterns).

Higher-level inputs are patterns formed by lower-level comparisons. They represent results or derivatives: match and miss per compared input parameter. So, number of parameters (variables) per pattern is selectively multiplied on each level. Match and miss between patterns is combined match | miss between their parameters. To maximize selection, search must be strictly incremental in distance, derivation, and composition over both. Which means a unique set of operations per level of search, hence a singular in “cognitive algorithm“.

Resulting hierarchy is a dynamically extended pipeline: terminated patterns are outputted for comparison on the next level, and new level is formed for pattern terminated by current top level. Which continues as long as system receives novel inputs. As distinct from autoencoders (current mainstay in unsupervised learning), there is no need for decoding: comparison is done on each level, whose output is also fed back to filter lower levels.

Autonomous cognition must start with analog inputs, such as video or audio. All symbolic data, including that in natural languages, is encoded by some prior cognitive process. To discover meaningful patterns in symbols, they must be decoded before being cross-compared. And the difficulty of decoding is exponential with the level of encoding, so hierarchical learning starting with raw sensory input is by far the easiest to implement (part 0).

To discover anything complex at “polynomial” cost, resulting patterns should also be hierarchical. In my model, each level of search adds a level of composition and a sub-level of differentiation to each input pattern. Higher-level search is selective per level of resulting pattern. Both composition and selection speeds-up search, to form longer range spatio-temporal and then conceptual patterns. Which also send feedback: filters and then motor action, to select lower-level inputs and locations with above-average additive predictive value (part 3).

Hierarchical approaches are common in unsupervised learning, and all do some sort of pattern recognition.
But none that I know of is strictly incremental in scope and complexity of discoverable patterns. Which is necessary for selection, thus scalability, vs. combinatorial explosion in search space. But selection is more expensive upfront and won’t pay in simple test problems. So, it’s not suitable for immediate experimentation. That’s probably why no one else seems to be working on anything sufficiently similar to my algorithm.



## Comparison to Artificial and Biological Neural Networks



ANN learns via some version of Hebbian “fire together, wire together” coincidence reinforcement. Normally, “neuron’s” inputs are weighed at synapses, then summed and thresholded into output. Final output also back- propagates to synapses and is combined with their last input to adjust the weights. This weight adjustment is learning. But prior summation degrades resolution of inputs, precluding any comparison between them (the inputs are recoverable, but why degrade and then spend a ton of computation restoring them).

This is an inherently coarse statistical method: inputs are summed within samples defined by initially random weights. These weights are trained into meaningful values by Stochastic Gradient Descent backpropagation, but this process is too expensive to scale without supervision or task-specific reinforcement.
CogAlg is comparison-first and resulting patterns are immediately meaningful. In other words, my initial feedback per pixel is simply a prior or adjacent pixel, which is infinitely finer-grained than backpropagation.

Currently the most successful method is CNN, which computes similarity as a product: input * kernel (weights), adjusted by some activation function. Again, kernels start with useless random weights and their adjustment is delayed (coarsified) by hidden layers. Human brain is born with a final number of neurons. Before birth, they have to develop and connect by processing random noise, and then learn by adjusting their connections. But in software, generating and deleting nodes that represent specific content should be far more efficient.

Conceptually, similarity is a common subset of comparands, but multiplication forms a superset, which exaggerates similarity. This is compensated by some activation function, but that’s another unprincipled and grossly inaccurate hack, which causes vanishing or exploding gradients. In the brain, multiplication is analog (cheap) and exaggeration adds resistance to noise. But noise cancellation should be separate from recognition, specific to local properties of noise, which should be learned and updated along with those of input itself.

ANN compute difference  (error) on their output layer, but not between hidden layers. This distinction is only justified in supervised learning, where we have some specific expectation. In unsupervised learning, all layers are equally unknown before the input. So, each layer should compute both similarity and difference: feedback to a lower layer. In a temporal input flow, such difference should update expectations. I think it is this delayed update that causes initialization bias in ANN, and much of similar confirmation bias in humans.

Also, both input and kernel in ANN are arrays, often 2D or higher. This is far more coarse, thus less potentially selective and efficient, than one-to-one comparison of laterally adjacent inputs in my algorithm. I use “level” vs. “layer” because my levels are not identical. Complexity of inputs and operations is incremental with elevation: initial inputs are pixels, higher-level inputs are patterns formed on lower levels. My inference is comparison that forms separate match and difference, and the difference is fed back to update level-wide filters.

Inspiration by the brain kept ANN research going for decades before they became useful. But their “neurons” are mere stick figures for real ones. Of course, most of complexity in a neuron is due to constraints of biology. Ironically, weighted summation in ANN may also be a no-longer needed compensation for such constraint:
neural memory requires dedicated connections (synapses), which makes individual input representation and comparison prohibitively expensive. But not anymore, we now have dirt-cheap random access memory.

Other biological constraints are very slow neurons, and the imperative of fast reaction for survival in the wild. Both favor fast though crude summation (vs. slower one-to-one comparison), at the cost of glacial training. Reaction speed became less important: modern society is quite secure, while continuous learning is far more important because of accelerating progress. Another constraint is noise: neurons often fire at random, so their spikes are summed to reduce noise. Which is not a good reason to degrade far more precise electronic signals.



## Comparison to Capsule Networks and Clustering



The nearest experimentally successful method is recently introduced “capsules”. Some similarities to CogAlg:
- capsules also output multi-variate vectors, “encapsulating” several properties, similar to my patterns
- these properties also include coordinates and dimensions, compared to compute differences and ratios
- these distances and proportions are also compared to find “equivariance” or affine transformations
- capsules also send direct feedback to lower layer (dynamic routing), vs. trans-hidden-layer backprop in ANN

But measure of similarity in CapsNet (“agreement” in dynamic routing) is still an unprincipled dot product, vs. additive compression in CogAlg. This is not specific to CapsNet, most current recognition algorithms, and seemingly the brain too,  select for dot product. To repeat, multiplication vastly exaggerates similarity. Which adds noise resistance, crucial for our horribly noisy brain, but that should be a separate noise-specific function.

Pure similarity is a common subset: the smaller of comparands. Which would be a compression of represented magnitude if a larger comparand is replaced with its difference to a smaller comparand. It’s a direct implication of information-theoretical compression-uber-alles, but no one else uses minimum as a measure of similarity. It’s not sufficient per se, the most basic working measure would probably be a deviation of relative match: minimum / miss - average minimum / miss, but minimum is unavoidable as a starting point.

CapsNets also use CNN layers to recognize basic features, which are then fed to capsule layers. But a truly general method must apply the same principles on all levels of processing, any differentiation should be learned rather than built-in. In current implementation, capsules contain only probability and pose variables.
My patterns have match instead of probability, a miss that includes pose variables, and selected properties of lower-level patterns. In my terms, Hinton’s equivariance is a match between misses: differences and distances.

All these variables are derived by incrementally complex comparison: a core operation on all levels of CogAlg. My hierarchy is a dynamic pipeline: pattern is displaced from its level by a miss to new input, then forwarded to existing or newly formed higher level. Which means that higher-level patterns include lower-level variables, as well as their derivatives. These derivatives are summed within a pattern, then evaluated for intra-pattern search and feedback. Thus, both a hierarchy of patterns per system, and a sub-hierarchy of variables per pattern, expand with experience. I think this is fundamentally superior to static design of CapsNet.

Another technique similar to mine is hierarchical clustering. But conventional clustering defines match as inverted difference between inputs. This is the opposite of ANN, which computes match but not coincident difference. And it’s also wrong: match is a common subset of comparands, distinct from and complementary to the difference between them. Both should be computed, because each has independent predictive value.

Some readers dismiss this outline as generalities, which lack a direct connection to my code.  But I don’t see a disconnect, beyond simple adaptation to 2D format of the input. Please enlighten me, I will owe you big time. Of course, current code only covers first-level processing, but higher levels will be incremental in nature. Others complain about lack of math, but CogAlg must be selectively incremental, and complex math is not.  There is no pseudocode or some high-level scheme. Again, incremental means starting from the lowest level, and I don’t know of any low-level code designed to maximize compression in a strictly incremental fashion. Any high-level architecture must be emergent: learned and forgettable depending on the input.



## Implementation



Any prediction has two components: what and where. We must have both: value of prediction = precision of what * precision of where. That “where” is currently neglected: statistical ML methods represent S-T dimensions with a significant lag, much more coarsely than the inputs themselves. Hence, precision of where (spans of and distances between patterns) is severely degraded, and so is predictive value of combined representations. There is no default degradation of positional information in my method.

My core algorithm is 1D: time only (part 4). Our space-time is 4D, but each of these dimensions can be mapped on one level of search. This way, levels can select input patterns that are strong enough to justify the cost of representing additional dimension, as well as derivatives (matches and differences) in that dimension.
Initial 4D cycle of search would compare contiguous inputs, analogously to connected-component analysis:

level 1 compares consecutive 0D pixels within horizontal scan line, forming 1D patterns: line segments.

level 2 compares contiguous 1D patterns between consecutive lines in a frame, forming 2D patterns: blobs.

level 3 compares contiguous 2D patterns between incremental-depth frames, forming 3D patterns: objects.

level 4 compares contiguous 3D patterns in temporal sequence, forming 4D patterns: processes.

Subsequent cycles would compare 4D input patterns over increasing distance in each dimension, forming longer-range discontinuous patterns. These cycles can be coded as implementation shortcut, or discovered by core algorithm itself, which can adapt to inputs of any dimensionality. “Dimension” here is a parameter that defines external sequence and distance among inputs. This is different from conventional clustering, which treats both external and internal parameters as dimensions.

However, average match in our space-time is presumably equal over all four dimensions. That means patterns defined in fewer dimensions will be biased by the angle of scanning, introducing artifacts. Hence, initial pixel comparison and inclusion into patterns should also be over 4D at once, or at least over 2D at once for images. This is a universe-specific extension of my core algorithm.

I have POC code for basic 1D core algorithm: https://github.com/boris-kz/CogAlg/blob/master/line_POC.py, am currently working on its adaptation to process images: https://github.com/boris-kz/CogAlg/blob/master/frame_blobs_draft.py , https://github.com/boris-kz/CogAlg/blob/master/frame_draft.py.
Initial testing could be recognition and automatic labeling of manually labeled images.

This algorithm will be organically extended to process colors, then video, then stereo video (from multiple confocal cameras).
For video, level 2 will process consecutive frames and derive temporal patterns, and levels 3 and higher will process discontinuous 2D + time patterns. It should also extend to any type and scope of data.

Suggestions and collaboration are most welcome, see last part of my intro on prizes.


