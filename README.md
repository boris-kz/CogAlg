CogAlg
======

This algorithm is a strictly bottom-up connectivity-based clustering, from pixels to eternity. It is directly derived from my definition of general intelligence: the ability to predict from prior / adjacent input. That includes planning, which is technically a self-prediction. Any prediction is interactive projection of known patterns, hence primary process must be pattern discovery (AKA unsupervised learning: an obfuscating negation-first term). This perspective is not novel, pattern recognition a main focus in ML, and a core of any IQ test. The problem I have with current ML is conceptual consistency. 

Pattern recognition is a default mode in Neural Nets, but they work indirectly, in a very coarse statistical fashion. Basic NNs, such as [multi-layer perceptrons](https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53), perform lossy stochastic chain-rule fitting. Each node outputs normalized sum of weighted inputs, then adjusts the weights in proportion to numeric similarity between input and output. In Deep Learning this adjustment is by backpropagated and decomposed error (inverse similarity) in the output layer, while in Hebbian Learning the adjustment is local, by input-output coincidence: binary version of their direct similarity. 

Modern ANNs combine such vertical training with lateral cross-correlation, within an input vector. CNN filters are designed to converge on edge-detection in initial layers. Edge detection means computing lateral gradient, by weighted pixel cross-comparison within kernels. Graph NNs embed lateral edges, representing similarity or/and difference between nodes, also produced by their cross-comparison. Popular [transformers](https://www.quantamagazine.org/researchers-glimpse-how-ai-gets-so-good-at-language-processing-20220414/) can be seen as a [variation of Graph NN](https://towardsdatascience.com/transformers-are-graph-neural-networks-bca9f75412aa). Their first step is self-attention: computing dot-product between query-key pairs within input vector. Dot-product serves as a measure of similarity in these pairs, so it's also a form of cross-comparison, but the measure is not principled.

So basic operation in both trained CNN and self-attention is what I call cross-comparison, but the former compute variance and the latter similarity. I think the difference is due their relative rarity in respective target data: sparse gradients in raw images and sparse similarities in compressed text. This rarity / inverse probability / surprise / negentropy, determines information content of the input. But almost all text actually describes generalized images and objects therein, so there should be a gradual transition between the two. In my scheme higher-level cross-comparison computes both difference and similarity, for differential clustering.

GNN, transformers, and Hinton's [Capsule Networks](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b) also have positional embeddings (as I use explicit coordinates). But they are still trained through destructive backprop: indiscriminate summation first, meaningful output-to-template comparison last. This primary summation degrades resolution of the whole learning process, exponentially with the number of layers. Hence, a ridiculous number of backprop cycles is needed to fit hidden layers into generalized representations (patterns) of the input. Most practitioners agree that this process is not very smart, we and evolution use it because it's relatively simple. It's also easy to parallelize, which is crucial for cell-based biology (and lazy human coding).

I think the process should be reversed: first cross-comparison of atomic inputs, then their summation into match-defined patterns (clusters): lateral [connectivity-based clustering](https://en.wikipedia.org/wiki/Cluster_analysis#Connectivity-based_clustering_(hierarchical_clustering)) vs. vertical statistical fitting in NNs. This cross-comp and clustering is recursively hierarchical, forming patterns of patterns and so on. Resulting compositional hierarchy is an indefinitely extended pipeline. 
As most interesting real-world systems, a connectivity cluster is defined by links / interactions between its nodes. Initial frame of reference here is space-time, but higher levels will reorder the input according to some critical parameters (eigenvalues). This re-framing will result in a version of [spectral clustering](https://en.wikipedia.org/wiki/Spectral_clustering). Feedback adjusts hyper-parameters to filter future inputs, vs. fitting them to templates. No training, just learning. 

Connectivity clustering is among the oldest approaches in ML, what I think makes my scheme different is:
- links are valued by similarity and variance, both derived by comparison between the nodes they connect, and potentiated by the overlap in the surround of these nodes (context).
- first-principles definition of similarity as compression, direct or inverse, while variance is negative, only valuable if borrowing from co-projected similarity.
- nested derivatives parameterize resulting clusters for higher-order cross-comp, incremental in range, derivation, and composition.
Such compressed encoding should be far more meaningful than huge flat weigh matrices in ANNs. But it’s very complex to consistently design and parallelize, precluding immediate trial and error that dominates ML. 

Below I describe the process in more detail, then extend comparisons to ANN and BNN. This is an open project: [WIKI](https://github.com/boris-kz/CogAlg/wiki), we need help with design and implementation. I pay for contributions or monthly if there is a track record, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).
Longer but partly obsolete introduction: <www.cognitivealgorithm.info>
This content is published under Creative Commons Attribution 4.0 International License.

###Outline of my approach

Initial clustering levels, positional resolution (macro) lags value resolution (micro) by one quantization order:

| Input                          | Comparison | Positional Resolution                        | Output                         | Conventionally known as            |
|-----------------------------------------|------------|----------------------------------------------|-----------------------------------------|------------------------------------|
| _unary_ intensity                         | AND        | _none,_ all in same coords                            | pixels of intensity                          | digitization                       |
| _integer_ pixels                          | SUB        | _binary:_ direction of comparison                            | blobs of gradient                       | edge detection, flood fill |
| _float:_ average blob params                     | DIV: compare blob params        | _integer:_ distance between blob centers                          | graphs of blobs              | connectivity-based clustering      |
| _complex:_ normalized graph params | LOG: compare graph params        | _float:_ distance between graph centers | hierarchical graphs | agglomerative clustering   |

And so on, higher levels should be added recursively. Such process is very complex and deeply structured, there is no way it could evolve naturally. Since the code is supposed to be recursive, testing before it is complete is almost useless. Which is probably why no one seems to work on such methods. But once the design is done, there is no need for interminable glacial and opaque training, my feedback only adjusts hyperparameters.

So, pattern is a cluster of matching items, where match is compression achieved by replacing elements with their derivatives, see “Comparison” section below. More commonly, pattern is a recurring set or order of elements, but to me this is 2nd order pattern. If the elements co-vary: don't match but their derivatives do, then the derivatives become elements of a higher-derivation pattern. 
Consistent process must start with cross-comp of adjacent atomic inputs: sensory data at the limit of resolution, such as pixels of video or equivalents in other modalities. Symbolic data is not a separate modality, just a generalized and encoded sensory data. The symbols must be decoded to discover meaningful patterns, which is exponentially difficult with the level of encoding. Thus, a start with raw sensory input is by far the easiest to implement (part 0).

This low-level process, directly translated into my code, seems like quite a jump from the generalities above. But it really isn’t, internally consistent pattern discovery must be strictly bottom-up, in the complexity of both inputs and operations. And there is no ambiguity at the bottom: initial predictive value that defines patterns is a match from cross-comparison among their elements, starting with pixels. So, I think my process is uniquely consistent with these definitions, please let me know if you see any discrepancy in either.

#### Comparison, more in part 1:

Basic comparison is inverse arithmetic operation between single-variable comparands, of incremental power: Boolean, subtraction, division, etc. Each order of comparison forms miss or loss: XOR, difference, ratio, etc., and match or similarity, which can be defined directly or as inverse deviation of miss. Direct match is compression of represented magnitude by replacing larger input with corresponding miss between the inputs: Boolean AND, the smaller input in comp by subtraction, integer part of ratio in comp by division, etc.

These direct similarity measures work if input intensity represents some stable physical property, which anti-correlates with variation. This is the case in tactile but not in visual input: brightness doesn’t correlate with inertia or invariance, dark objects are just as stable as bright ones. Thus, initial match in vision should be defined indirectly, as inverse deviation of variation in intensity. 1D variation is difference, ratio, etc., while multi-D comparison has to combine them into Euclidean distance and gradient, as in common edge detectors.

#### Patterns, more in part 2:

Cross-comparison among patterns forms match and miss per parameter, as well as dimensions and distances: external match and miss (these are separate parameters: total value = precision of what * precision of where). Comparison is limited by max. distance between patterns. Overall hierarchy has incremental dimensionality: search levels ( param levels ( pattern levels)).., and pattern comparison is selectively incremental per such level. This is hard to explain in NL, please see the code, starting with [line_Ps](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_Ps.py) and [line_PPs](https://github.com/boris-kz/CogAlg/blob/master/line_1D_alg/line_PPs.py).
  
Resulting matches and misses are summed into lateral match and miss per pattern. Proximate input patterns with above-average match to their nearest neighbors are clustered into higher-level patterns. This adds two pattern levels: of composition and derivation, per level of search. Conditional cross-comp over incremental range and derivation, among the same inputs, may also add sub-levels in selected newly formed patterns. On a pixel level, incremental range is using larger kernels, and incremental derivation starts with using Laplacian. 

#### Feedback: attention, imagination, action, more in part 3 (tentative, not in the code)

Higher-level feedback will adjust filters, starting with average match, then ave per parameter derived by deeper cross-comp. More precisely, these should be co-averages: values coincident with an average value of combined higher-level param. There are also positional or external filters, starting with pixel size and kernel size, which determine external dimensions of the input. Quantization (bit, integer, float..) of internal and external filters corresponds to the order of comparison. The filters are similar to hyperparameters in Neural Nets, with the same values across a level. The equivalent to weight matrix are links (edges) between nodes of a graph, but they are lateral vs. implicitly vertical when formed via backprop or [Hebbian learning](https://data-flair.training/blogs/learning-rules-in-neural-network/#:~:text=The%20Hebbian%20rule%20was%20the,of%20nodes%20of%20a%20network.&text=For%20neurons%20operating%20in%20the,weight%20between%20them%20should%20decrease.) in NNs.

In a broader frame of reference, the above-mentioned external filters will define source locations for selective input to higher-level patterns. This is similar to conventionally understood attention, and ultimately decision making. And these locations can be projected vs. actually observed, generating input for imagination and hypothetical reasoning.


#### Hierarchy, part 4 is out of date:

There is a single global hierarchy: feedforward inputs and feedback filters pass through the same levels of search and composition. Each higher level is a nested hierarchy, with depth proportional to elevation, but sub-hierarchies are unfolded sequentially. That’s why I don’t have many diagrams: they are good at showing relations in 2D, but I have a simple 1D sequence of levels. Nested sub-hierarchies are generated by the process itself, depending on elevation in a higher-order hierarchy. That means I can’t show them in a generic diagram. 

Brain-inspired schemes have separate sensory and motor hierarchies, in mine they combined into one. The equivalent of motor patterns in my scheme are positional filter patterns, which ultimately move the sensor. The first level is co-located sensors: targets of input filters, and more coarse actuators: targets of positional filters. I can think of two reasons they are separated in the brain: neurons and axons are unidirectional, and training process has to take the whole hierarchy off-line. Neither constraint applies to my scheme.

Final algorithm will consist of first-level operations + recursive increment in operations per level. The latter is a meta-algorithm that extends working level-algorithm, to handle derivatives added to current inputs. So, the levels are: 1st level: G(x), 2nd level: F(G)(x), 3rd level: F(F(G))(x).., where F() is the recursive code increment. Resulting hierarchy is a pipeline: patterns are outputted to the next level, forming a new level if there is none. As long as there are novel inputs, higher levels will discover longer-range spatio-temporal and then conceptual patterns. 
 
Please see [system diagram](https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/Whole-system%20hierarchy.png). 

Some notes:
- There should be a unique set of operations added per level, hence a singular in “cognitive algorithm”.
- Core design must be done theoretically: generality requires large upfront investment in process complexity, which makes it a huge overkill for any specific task. That’s one reason why such schemes are not explored.
- Many readers note disconnect between abstractions in this outline, and the amount of detail in current code. That’s because we are in space-time continuum: search must follow proximity in each dimension, which requires specific processing. It’s not specific to vision, the process is roughly the same for all raw modalities. 
- Another complaint is that I don't use mathematical notation, but it doesn't have the flexibility to express deeply conditional process. Math is not separable from logic here.
- Most people who aspire to work on AGI think in terms behavior and robotics. I think this is far too coarse to make progress, the most significant mechanisms are on the level of perception. Feedforward (perception) must drive feedback (action), not the other way around.
- Other distractions are supervision and reinforcement. These are optional task-specific add-ons, core cognitive process is unsupervised pattern discovery, and main problem here is scaling in complexity.
- Don’t even start me on chatbots.  


### Comparison to Artificial and Biological Neural Networks


All unsupervised learning is some form of pattern discovery, where patterns are some kind of similarity clusters. There are two fundamentally different ways to cluster inputs: centroid-based and connectivity-based. All [statistical learning](https://en.wikipedia.org/wiki/Statistical_learning_theory), including Neural Nets, is best understood as distributed [centroid-based clustering](https://en.wikipedia.org/wiki/Cluster_analysis#Centroid-based_clustering). Centroid is whatever the model fits to, not necessarily a single value. Template line in linear regression can be considered one-dimensional centroid, and the whole training set a multi-dimensional centroid. 

That usually means training [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) or Transformer to perform some sort of edge-detection or cross-correlation (same as my cross-comparison but the former terms lose meaning on higher levels of search). But CNN operations are initially random, while my process is designed for cross-comp from the start. This is why it can be refined by my feedback, updating the filters, which is far more subtle and selective than weight training by backprop. So, I have several problems with basic process in ANN:

- Vertical learning (via feedback of error) takes tens of thousands of cycles to form accurate representations. That's because summation per layer degrades positional input resolution. With each added layer, the output that ultimately drives learning contains exponentially smaller fraction of original information. My cross-comp and clustering is far more complex per level, but the output contains all information of the input. Lossy selection is only done on the next level, after evaluation per pattern (vs. before evaluation in statistical methods). 

- Both initial weights and sampling that feeds [SGD](https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31) (Stochastic Gradient Descent) are randomized. Also driven by random variation are [RBMs](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine), [GANs](https://en.wikipedia.org/wiki/Generative_adversarial_network), [VAEs](https://golden.com/wiki/Variational_autoencoder), etc. But randomization is antithetical to intelligence, it's only useful in statistical methods because they merge inputs with weights irreversibly. Thus, any non-random initialization and variation will introduce bias. All input modification in my scheme is via hyper-parameters, stored separately and then used to normalize (remove bias) inputs for comparison to inputs formed with different-value hyper-parameters. 

- SGD minimizes error (top-layer miss), which is quantitatively different from maximizing match: compression. And that error is w.r.t. some specific template, while my match is summed over all past input / experience. The “error” here is plural: lateral misses (differences, ratios, etc.), computed by cross-comparison within a level. All inputs represent environment and have positive value. But then they are packed (compressed) into patterns, which have different range and precision, thus different relative value per relatively fixed record cost.

- Representation in ANN is fully distributed, similar to the brain. But the brain has no alternative: there is no substrate for local memory or program in neurons. Computers have RAM, so parallelization is a simple speed vs. efficiency trade-off, useful only for complex semantically isolated nodes. Such nodes are patterns, encapsulating a set of co-derived “what” and “where” parameters. This is similar to neural ensemble, but parameters that are compared together should be localized in memory, not distributed across a network.

More basic neural learning mechanism is Hebbian, though it is rarely used in ML. Conventional spiking version is that weight is increased if the synapse often receives a spike just before the node fires, else the weight is decreased. But input and output don't have to be binary, the same logic can be applied to scalar values: the weight is increased / decreased in proportion to some measure of similarity between its input and following output of the node. That output is normalized sum of all inputs, or their centroid.

Such learning is local, within each node. But it's still a product of vertical comparison: centroid is a higher order of composition than individual inputs. This comparison across composition drives all statistical learning, but it destroys positional information at each layer. Compared to autoencoders (main backprop-driven unsupervised learning technique), Hebbian learning lacks the decoding stage, as does the proposed algorithm. Decoding decomposes hidden layers, to equalize composition orders of output and compared template.

Inspiration by the brain kept ANN research going for decades before they became useful. Their “neurons” are mere stick figures, but that’s not a problem, most of neuron’s complexity is due to constraints of biology. The problem is that core mechanism in ANN, weighted summation, may also be a no-longer needed compensation for such constraints: neural memory requires dedicated connections. That makes representation and cross-comparison of individual inputs very expensive, so they are summed. But we now have dirt-cheap RAM.

Other biological constraints are very slow neurons, and the imperative of fast reaction for survival in the wild. Both favor fast though crude summation, at the cost of glacial training. Reaction speed became less important: modern society is quite secure, while continuous learning is far more important because of accelerating progress. Summation also reduces noise, which is very important for neurons that often fire at random, to initiate and maintain latent connections. But that’s irrelevant for electronic circuits.

I see no way evolution could produce proposed algorithm, it is extremely limited in complexity that can be added before it is pruned by natural selection. And that selection is for reproduction, while intelligence is distantly instrumental. The brain evolved to guide the body, with neurons originating as instinctive stimulus-to-response converters. Hence, both SGD and Hebbian learning is fitting, driven by feedback of action-triggering weighted input sum. Pattern discovery is their instrumental upshot, not an original purpose.

Uri Hasson, Samuel Nastase, Ariel Goldstein reach a similar conclusion in “[Direct fit to nature: an evolutionary perspective on biological and artificial neural networks](https://www.cell.com/neuron/fulltext/S0896-6273(19)31044-X)”: “We argue that neural computation is grounded in brute-force direct fitting, which relies on over-parameterized optimization algorithms to increase predictive power (generalization) without explicitly modeling the underlying generative structure of the world. Although ANNs are indeed highly simplified models of BNNs, they belong to the same family of over-parameterized, direct-fit models, producing solutions that are mistakenly interpreted in terms of elegant design principles but in fact reflect the interdigitation of ‘‘mindless’’ optimization processes and the structure of the world.”



### Atomic comparison: quantifying match and miss between variables



First, we need to quantify predictive value. Algorithmic information theory defines it as compressibility of representation, which is perfectly fine. But compression is currently computed only for sequences of inputs, while I think a logical start is analog input digitization: a rock bottom of organic compression hierarchy. The next level is cross-comparison among resulting pixels, commonly known as edge detection, and higher levels will cross-compare resulting patterns. Partial match computed by comparison is a measure of compression.

Partial match between two variables is a complementary of miss, in corresponding power of comparison: 
- Boolean match is AND and miss is XOR (two zero inputs form zero match and zero miss), 
- comparison by subtraction increases match to a smaller comparand and reduces miss to a difference,
- comparison by division increases match to min * integer part of ratio and reduces miss to a fractional part
(direct match works for tactile input. but reflected-light in vision requires inverse definition of initial match)

In other words, match is a compression of larger comparand’s magnitude by replacing it with miss. Which means that match = smaller input: a common subset of both inputs, = sum of AND between their uncompressed (unary code) representations. Ultimate criterion is recorded magnitude, rather than bits of memory it occupies, because the former represents physical impact that we want to predict. The volume of memory used to record that magnitude depends on prior compression, which is not an objective parameter. 

Given incremental complexity, initial inputs should have binary resolution and implicit shared coordinate (being a macro-parameter, resolution of coordinate lags that of an input). Compression of bit inputs by AND is well known as digitization: substitution of two lower 1 bits with one higher 1 bit. Resolution of coordinate (input summation span) is adjusted by feedback to form integers that are large enough to produce above-average match.

Next-order compression can be achieved by comparison between consecutive integers, distinguished by binary (before | after) coordinate. Basic comparison is inverse arithmetic operation of incremental power: AND, subtraction, division, logarithm, and so on. Additive match is achieved by comparison of a higher power than that which produced comparands: comparison by AND will not further compress integers previously digitized by AND.

Rather, initial comparison between integers is by subtraction, resulting difference is miss, and smaller input is absolute match. Compression of represented magnitude is by replacing i1, i2 with their derivatives: match (min) and miss (difference). If we sum each pair, inputs: 5 + 7 -> 12, derivatives: match = 5 + miss = 2 -> 7. Compression by replacing = match: 12 - 7 -> 5. Difference is smaller than XOR (non-zero complementary of AND) because XOR may include opposite-sign (opposite-direction) bit pairs 0, 1 and 1, 0, which are cancelled-out by subtraction.

Comparison by division forms ratio, which is a magnitude-compressed difference. This compression is explicit in long division: match is accumulated over iterative subtraction of smaller comparand from remaining difference. In other words, this is also a comparison by subtraction, but between different orders of derivation. Resulting match is smaller comparand * integer part of ratio, and miss is final reminder or fractional part of ratio.

A ratio can be further compressed by converting to a radix | logarithm, and so on. But computational costs may grow even faster. Thus, power of comparison should increase only for inputs sufficiently compressed by lower power: AND for bit inputs, SUB for integer inputs, DIV for pattern inputs, etc. Actual compression depends on input and on resolution of its coordinate: input | derivative summation span. We can’t control the input, so average match is adjusted via coordinate resolution.

But the costs of operations and incidental sign, fraction, irrational fraction, etc. may grow even faster. To justify the costs, the power of comparison should only increase in patterns of above-average match from prior order of comparison: AND for bit inputs, SUB for integer inputs, DIV for pattern inputs, etc. Inclusion into such patterns is by relative match: match - ave: past match that co-occurs with average higher-level match.

Match value should be weighted by the correlation between input intensity and its stability: mass / energy / hardness of an observed object. Initial input, such as reflected light, is likely to be incidental: such correlation is very low. Since match is the magnitude of smaller input, its weight should also be low if not zero. In this case projected match consists mainly of its inverse component: match cancellation by co-derived miss, see below. 

The above discussion is on match from current comparison, but we really want to know projected match to future or distant inputs. That means the value of match needs to be projected by co-derived miss. In comparison by subtraction, projected match = min (i1, i2) * weight (fractional) - difference (i1, i2) / 2 (divide by 2 because the difference only reduces projected input, thus min( input, projected input), in the direction in which it is negative. It doesn’t affect min in the direction where projected input is increasing). 


#### quantifying lossy compression


There is a general agreement that compression is a measure of similarity, but no one seems to apply it from the bottom up, the bottom being single scalars. Also, any significant compression must be lossy. This is currently evaluated by perceived similarity of reconstructed input to the original input, as well as compression rate. Which is very coarse and subjective. Compression in my level of search is lossless, represented by match on all levels of pattern. All derived representations are redundant, so it’s really an expansion vs. compression overall.  

The lossy part comes after evaluation of resulting patterns on the next level of search. Top level of patterns is cross-compared by default, evaluation is per lower level: of incremental derivation and detail in each pattern. Loss is when low-relative-match buffered inputs or alternative derivatives are not cross-compared. Such loss is quantified as the quantity of representations in these lower levels, not some subjective quality. 

Compression also depends on resolution of coordinate (input summation span), and of input magnitude. Projected match can be kept above system’s average by adjusting corresponding resolution filters: most significant bits and least significant bits of both coordinate and magnitude.


### Implementation


Any prediction has two components: what and where. We must have both: value of prediction = precision of what * precision of where. That “where” is currently neglected: statistical ML methods represent coordinates much more coarsely than the inputs. Hence, precision of where (spans of and distances between patterns) is degraded, and so is predictive value of combined representations. That's not the case here because my top-level patterns (multi-dimensional blobs) are contiguous.

Core algorithm is 1D: time only. Our space-time is 4D, and average match is presumably equal over all dimensions. That means patterns defined in fewer dimensions will be only slices of actual input, fundamentally limited and biased by the angle of scanning / slicing. Hence, initial pixel comparison should also be over 4D at once, or at least over 3D for video and 2D for still images. This full-D-cycle level of search is a universe-specific extension of core algorithm. The dimensions should be discoverable by the core algorithm, but coding it in is much faster. 

This repository currently has three versions of 1st D-cycle, each analogous to connected-component analysis: [1D line alg](https://github.com/boris-kz/CogAlg/tree/master/line_1D_alg), [2D frame alg](https://github.com/boris-kz/CogAlg/tree/master/frame_2D_alg), and [3D video alg](https://github.com/boris-kz/CogAlg/tree/master/video_3D_alg).
Subsequent cycles will compare full-D-terminated input patterns over increasing distance in each dimension, forming discontinuous patterns of incremental composition and range.
“Dimension” here defines external sequence and distance among inputs. This is different from conventional clustering, which treats both external and internal parameters as dimensions. 

Complete hierarchical algorithm will have two-level code: 
- 1st level algorithm: contiguous cross-comparison over full-D cycle, plus feedback to adjust most and least significant bits of the input. 
- Recurrent increment in complexity, extending current-level alg to next-level alg. This increment will account for increasing internal complexity of input patterns on higher levels, unfolding them for cross-comparison and re-folding results for evaluation and feedback.

Initial testing could be on recognition of labeled images, but video or stereo video should be much better. We will then add colors, maybe audio and text. 

For more detailed account of current development see [WIKI](https://github.com/boris-kz/CogAlg/wiki).

Suggestions and collaboration are most welcome, see [CONTRIBUTING](https://github.com/boris-kz/CogAlg/blob/master/CONTRIBUTING.md).


