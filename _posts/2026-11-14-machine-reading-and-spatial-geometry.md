---
layout: post
title: "Machine reading and Spatial Geometry"
date: 2026-11-14
tags: [notes]
---

In my last post on BPE Tokenizers, I discussed how machines break down words that they have never seen before. But knowing new words isn’t the same as using them in the right context. One possible way for machines to learn context is by calculating full-vocabulary probabilities across billions of words, but that would be a computational nightmare. So, researchers introduced Skip-Gram with Negative Sampling (SGNS) - a smart approach that learns word vectors by training a model to distinguish authentic word pairs from artificially generated "negative" ones.

Traditional language models work by taking a sequence of context words and then trying to predict the single next word. Skip-gram reverses this entire process. Instead of feeding the model a full sentence and asking it to guess the next word, Skip-gram is given a center word (target) and is asked to predict the surrounding neighbourhood of words (context window). Consider the example: “I want to have warm coffee this morning.” Here, “coffee” is the target word and we will be using a size 2 sliding window. 
Target: coffee
Context: [“have”, “warm”, “this”, “morning”]
The model creates training pairs directly from this window:
(coffee, have)
(coffee, warm)
(coffee, this)
(coffee, morning)
Inverting the problem works because the model does not have to predict all of its surrounding neighbours at once. It evaluates each target-context pair independently, ignoring word order so that words that share similar neighbours naturally align in vector space. 

When the model looks at a training pair like (coffee, warm), it computes the dot product of the vector “coffee” and the vector for “warm”. If the value of the dot product is high, then the vectors are pointing in the same direction in geometric space. If they are unrelated, the number is low. The raw dot product that we have now calculated isn’t probability. To calculate how confident the model is about “warm” compared to other words in the vocabulary, the model runs a formula called Softmax. To calculate the probability of “warm”, the model must take the exponential of its score and divide it by the sum of exponentials of all words in the entire dictionary (|V|):
$$
\text{Probability}(\text{"warm"} \mid \text{"coffee"}) = \frac{e^{\text{score}(\text{coffee, warm})}}{\sum_{w=1}^{|V|} e^{\text{score}(\text{coffee, } w)}}
$$
This is a major bottleneck because if my vocabulary has 1,000,000 words, then the model is required to calculate the exponentials of this huge denominator  of all 1,000,000 words for every single pair. This would be computationally infeasible. 

To bypass this challenge, researchers built a conceptual shortcut: Negative Sampling. Instead of asking the model “Out of 1,000,000 words in the vocabulary, which word is the correct neighbour?” negative sampling reframes the problem into a simple True vs. Fake Quiz. For every pair, instead of scoring the entire dictionary, the model has to only evaluate a few words. For example,
Target: coffee | Neighbour: warm -> Label: TRUE(1)
Target: coffee | Neighbour: tree -> Label: FAKE(0)
Target: coffee | Neighbour: airport -> FAKE(0)
Target: coffee | Neighbour: museum -> FAKE(0)
Target: coffee | Neighbour: dog -> FAKE(0)
Target: coffee | Neighbour: woman -> FAKE(0)
Instead of updating a huge matrix across millions of classes for every single step, the model adjusts the weights slightly so that only the true neighbour (warm) gets high scores and the fake neighbours (tree, airport, museum, dog, woman) get low scores.

One important caveat is that the model does not pre-build a permanent list of words that can never go together. In daily conversations, almost any of the above words can come together in a sentence. For example, “I drank my coffee at the airport” has “airport” as a neighbour of “coffee”. Every time the model sees the training pair (coffee, warm), the training pipeline looks into the master dictionary and grabs 5 other words randomly. For that sentence, these 5 random words are assigned a label FAKE(0). However, in a later step, if any of those random words actually show up in a real sentence, then it gets labeled TRUE(1). 

To summarize, Skip-gram with Negative Sampling (SGNS) gave machines a spatial map of how words relate in context without having to run through the entire vocabulary. While SGNS uses static vectors, modern LLMs use dynamic, contextual embeddings as soon as they see a sentence.



