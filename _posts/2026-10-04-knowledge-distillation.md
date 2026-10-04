---
layout: post
title: "Demystifying Knowledge Distillation: Math, Gradients, and NumPy Implementations"
date: 2026-10-04
tags: [notes]
---

Knowledge distillation is a concept where a smaller "student" model learns from a larger, pre-trained "teacher" model. Instead of training the student on hard labels like [0,0,1], the student is trained to match the soft probabilities produced by the teacher model, like [0.067, 0.1345, 0.7985]. These soft targets contain "dark knowledge" -- richer information about class similarities, such as a cat looking more like a dog than a car.

To generate soft probabilities, we take the raw outputs of the final layer of a neural network, called logits, and pass them through a temperature-scaled Softmax function. The temperature T controls how sharp or flat the probability distribution becomes. Standard softmax (T=1) produces highly confident "peaked" predictions, hiding the dark knowledge in near-zero probabilities. Raising T flattens the distribution and exposes the relative relationships across non-target classes so the student can learn more efficiently.

## Temperature Scaling and Dark Knowledge

To implement temperature-scaled Softmax, we modify the standard Softmax formula by dividing the raw logits by T before exponentiating:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

In NumPy, we subtract the max logit for numerical stability:

```python
import numpy as np

def softmax_with_temperature(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    scaled_logits = (logits - np.max(logits, axis=-1, keepdims=True)) / T
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
```

To see temperature scaling in action, consider a teacher model outputting logits v = [10.0, 5.0, -2.0] for [Cat, Dog, Car]:

| Temperature (T) | Soft Probabilities | Behavior |
|---|---|---|
| T = 1.0 | [0.9933, 0.0067, 0.0000] | Sharp: ~99.3% confident in Cat. Dog and Car are buried. |
| T = 2.0 | [0.9168, 0.0753, 0.0079] | Dog starts to show a signal over Car. |
| T = 5.0 | [0.6033, 0.2220, 0.0544] | Dog holds 22.2%, clearly a plausible second choice. |
| T = 20.0 | [0.4228, 0.3326, 0.2446] | Flat: Dog is much closer to Cat than Car is. |

At high temperatures, the teacher reveals the structural landscape of the dataset -- the precise dark knowledge the student needs to mimic.

## Gradient Dynamics and the T² Scaling Factor

Once we obtain soft probability distributions p (teacher) and q (student), we measure how far apart they are using Soft Cross-Entropy. However, turning up temperature T introduces a subtle trap: it destroys the magnitude of our gradients during backpropagation.

### The Vanishing Gradient Problem

The standard soft cross-entropy loss is:

$$\mathcal{L}_{\text{soft\_raw}} = -\sum_i p_i \log(q_i)$$

When we scale logits by T, applying the chain rule gives:

$$\frac{\partial \mathcal{L}_{\text{soft\_raw}}}{\partial z} = \frac{1}{T}(q - p)$$

As T increases, two things happen simultaneously:

1. The difference (q - p) shrinks because higher temperatures flatten both distributions.
2. The entire expression is multiplied by 1/T.

Because (q - p) scales roughly as O(1/T), the gradient shrinks at O(1/T²). If you raise T to 20 or 100, your gradients vanish to near-zero and the student stops learning.

### The Fix: Multiplying Loss by T²

Geoffrey Hinton et al. scaled the soft cross-entropy loss by T²:

$$\mathcal{L}_{\text{soft}} = -T^2 \sum_i p_i \log(q_i)$$

Taking the derivative, one factor of T cancels out:

$$\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z} = T^2 \cdot \frac{1}{T}(q - p) = T(q - p)$$

This T² multiplier restores scale invariance, keeping gradient magnitudes steady across different values of T.

### Numerical Proof in NumPy

```python
import numpy as np

def softmax_with_temp(logits: np.ndarray, T: float) -> np.ndarray:
    scaled = (logits - np.max(logits, axis=-1, keepdims=True)) / T
    exp_logits = np.exp(scaled)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

v = np.array([2.0, 5.0, 1.0])
z = np.array([1.0, 2.0, 0.0])

print(f"{'Temp (T)':<10} | {'Unscaled Grad (q - p)/T':<25} | {'Scaled Grad T * (q - p)':<25}")
print("-" * 65)

for T in [1.0, 5.0, 20.0, 100.0]:
    p = softmax_with_temp(v, T)
    q = softmax_with_temp(z, T)
    grad_unscaled = (q - p) / T
    grad_scaled = T * (q - p)
    print(f"T = {T:<8} | {str(np.round(grad_unscaled, 5)):<25} | {str(np.round(grad_scaled, 4)):<25}")
```

Output:

```
Temp (T)   | Unscaled Grad (q - p)/T    | Scaled Grad T * (q - p)
-----------------------------------------------------------------
T = 1.0    | [ 0.19812 -0.271    0.07288] | [ 0.1981 -0.271   0.0729]
T = 5.0    | [ 0.01085 -0.01974  0.00889] | [ 0.2714 -0.4935  0.2222]
T = 20.0   | [ 0.00059 -0.00115  0.00056] | [ 0.2366 -0.4616  0.225 ]
T = 100.0  | [ 0.00002 -0.00004  0.00002] | [ 0.2252 -0.4481  0.2229]
```

Without T² scaling, the gradient collapses from 0.1981 down to 0.00002 as T increases. With T² scaling, the gradient stabilizes around [0.225, -0.448, 0.222] regardless of temperature.
