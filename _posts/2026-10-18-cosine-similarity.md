---
layout: post
title: "Cosine Similarity"
date: 2026-10-18
categories: [Machine Learning, NLP]
tags: [notes]
---

One-hot encodings represent each word as a sparse binary vector where only one position is 1 and the rest are 0. The problem with one-hot encodings is that every pair of words is equally distant from each other. There is no notion of semantic similarity.

Word embeddings fix this by placing words in a geometric space where meaning is encoded as direction and distance. Words that appear in similar contexts end up with similar vectors.

But even with embeddings, Euclidean distance has a problem. High-frequency words like "the" appear in almost every context, which inflates their embedding norm. This makes them appear far from most other words even if they point in the same direction.

Cosine similarity solves this by measuring only the angle between two vectors, ignoring magnitude entirely:

$$\text{Sim}(u, v) = \frac{u \cdot v}{\|u\| \cdot \|v\|}$$

This is equivalent to L2-normalizing both vectors and taking their dot product. The result is 1 for identical directions, 0 for orthogonal, and -1 for opposite.

```python
import numpy as np

vocab = ["king", "man", "queen", "woman", "apple"]

E = np.array([
    [0.50, 0.70, 0.20, 0.90],  # king
    [0.40, 0.80, 0.10, 0.30],  # man
    [0.55, 0.25, 0.85, 0.95],  # queen
    [0.45, 0.35, 0.80, 0.35],  # woman
    [0.05, 0.10, 0.05, 0.05]   # apple
])

norms = np.linalg.norm(E, axis=1, keepdims=True)
E_norm = E / norms
similarity_matrix = E_norm @ E_norm.T

print("Pairwise Similarity Matrix shape:", similarity_matrix.shape)
print(np.round(similarity_matrix, 2))
```

**Output:**
```text
[[1.   0.88 0.83 0.72 0.9 ]
 [0.88 1.   0.59 0.65 0.96]
 [0.83 0.59 1.   0.92 0.76]
 [0.72 0.65 0.92 1.   0.83]
 [0.9  0.96 0.76 0.83 1.  ]]
```
