---
layout: post
title: "Cosine Similarity"
date: 2026-10-18
categories: [Machine Learning, NLP]
tags: [notes]
---

In this post, we will be understanding the geometry of word embeddings. Word embeddings allow us to create a geometric landscape that shape the meaning based on where we are standing and in which direction. Earlier, one-hot encoding was used to obtain a vector representation of an array. For example, if we had an array A = [cat, dog, lion, elephant], then cat would be encoded as [1,0,0,0], dog as [0,1,0,0], lion as [0,0,1,0] and elephant as [0,0,0,1]. It is impossible to find the angle of similarity in this case because 0 multiplied with any number would give 0. So the similarity would be 0 no matter how probable the next word is. There would be no difference between the words 'cat' and 'dog' ,just like there would be no difference between 'cat' and 'aeroplane'.

An embedding space on the other hand would be a vector of similarities, like [0.11, -.58, 0.67, 0.43]. It represents semantic traits along axes we don't explicitly name. Words with similar meanings, like cat and dog, end up close to each other because they were found together in many texts.

The very first technique we think of when we hear measuring distance between two points is the Euclidean distance, i.e.,

$$d(u,v) = \|u - v\|_2 = \sqrt{\sum (u_i - v_i)^2}$$

Consider the sentence "The cat chased the mouse across the threshold of the kitchen, while the dog watched the scene from the porch of the house near the river.". This sentence already contains 9 "the" words, but words like "cat", "mouse", "dog" appear just once. So, the algorithm would think that there is a high similarity between "the" and any other word. So the model ends up learning nothing useful. Let's take a small example to illustrate this challenge:

"the" (A hyper-frequent filler word that appears everywhere):

$$v_{\text{the}} = [2.5, 3.0, 4.0]$$

"house" (A specific, content-heavy noun):

$$v_{\text{house}} = [0.5, 0.6, 0.8]$$

Let's find the "length" (magnitude) of each word's vector using the L2 norm formula ($\sqrt{\sum x^2}$):

- Length of "the":
  $$\|v_{\text{the}}\|_2 = \sqrt{2.5^2 + 3.0^2 + 4.0^2} = \sqrt{6.25 + 9.0 + 16.0} = \sqrt{31.25} \approx 5.59$$

- Length of "house":
  $$\|v_{\text{house}}\|_2 = \sqrt{0.5^2 + 0.6^2 + 0.8^2} = \sqrt{0.25 + 0.36 + 0.64} = \sqrt{1.25} \approx 1.12$$

Because of this huge value of "the", the model will think that there is no relation between "house" and "the". Cosine similarity, on the other hand, strips away the frequency-driven size of the vector and only looks at what direction the word is pointing.

Cosine similarity does not consider the magnitude and measures only the angle between two vectors.

$$\text{Sim}(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

If two vectors are unrelated, then the angle would be 90 degree. If they are exact opposites, it would be -1.0 and if they were related, it would be 1.0. By L2-normalizing every vector beforehand, the dot-product of two normalized vectors becomes their cosine similarity.

## NumPy Implementation

```python
import numpy as np

# 1. Mock 4-dimensional embeddings for a 5-word vocabulary
vocab = ["king", "man", "queen", "woman", "apple"]

E = np.array([
    [0.50, 0.70, 0.20, 0.90],  # king
    [0.40, 0.80, 0.10, 0.30],  # man
    [0.55, 0.25, 0.85, 0.95],  # queen
    [0.45, 0.35, 0.80, 0.35],  # woman
    [0.05, 0.10, 0.05, 0.05]   # apple
])

# 2. Matrix-wide L2 Normalization (keepdims=True ensures proper broadcasting)
norms = np.linalg.norm(E, axis=1, keepdims=True)
E_norm = E / norms

# 3. Pairwise Cosine Similarity Matrix (5x5) via single matrix multiplication
similarity_matrix = E_norm @ E_norm.T

print("Pairwise Similarity Matrix shape:", similarity_matrix.shape)
print(np.round(similarity_matrix, 2))
```

In this post, I have considered a toy example to illustrate the calculation of vectors of an array. In reality, before training, a word embedding model assigns a random vector to each word. The model is given a massive task to go through billions of text documents. Every time it sees the word "house", it updates the vector. It does the same for "the" as well. However, since we are normalizing the vectors, the number of "the" in a sentence does not matter to the model.
