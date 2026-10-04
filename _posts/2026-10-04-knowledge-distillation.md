---
layout: post
title: "Demystifying Knowledge Distillation: Math, Gradients, and NumPy Implementations"
date: 2026-10-04
categories: [Machine Learning, Deep Learning]
tags: [notes]
---

> **TL;DR:** Knowledge Distillation transfers "dark knowledge" from a teacher model to a student model using soft probabilities. In this post, we unpack the exact math behind temperature scaling ($T$), prove why loss must be scaled by $T^2$ to avoid vanishing gradients, and verify how high-temperature soft cross-entropy aligns with zero-mean centered MSE in pure NumPy.

---

## 1. What is Knowledge Distillation?

Knowledge distillation is a technique where a smaller "student" model learns from a larger, pre-trained "teacher" model...

*(Include your introductory explanation comparing hard vs. soft labels here)*

---

## 2. Temperature Scaling and "Dark Knowledge"

To generate soft probabilities, we modify the standard Softmax function by introducing a **Temperature ($T$)** hyperparameter:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

### NumPy Implementation

```python
import numpy as np

def softmax_with_temperature(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    scaled_logits = (logits - np.max(logits, axis=-1, keepdims=True)) / T
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

## 3. Gradient Dynamics & The $T^2$ Scaling Factor

Once we obtain soft probability distributions $p$ (teacher) and $q$ (student), we measure how far apart they are using Soft Cross-Entropy. However, simply turning up temperature $T$ introduces a subtle mathematical trap: **it destroys the magnitude of our gradients during backpropagation.**

### The Vanishing Gradient Problem

The standard soft cross-entropy loss without temperature scaling is defined as:

$$\mathcal{L}_{\text{soft\_raw}} = -\sum_i p_i \log(q_i)$$

When we scale logits by temperature $T$, every logit $z_i$ is replaced by $\frac{z_i}{T}$. Applying the chain rule to compute the gradient with respect to the student’s unscaled logits $z$ gives:

$$\frac{\partial \mathcal{L}_{\text{soft\_raw}}}{\partial z} = \frac{1}{T}(q - p)$$

As temperature $T$ increases, two things happen simultaneously:
1. The difference between probabilities $(q - p)$ shrinks because higher temperatures flatten both distributions.
2. The entire expression is multiplied by $\frac{1}{T}$.

Because $(q - p)$ itself scales roughly on the order of $\mathcal{O}(1/T)$ at high temperatures, the unscaled gradient shrinks at a rate of $\mathcal{O}(1/T^2)$. If you raise $T$ to $20$ or $100$, **your gradients vanish to near-zero**, and the student model stops learning!

### The Fix: Multiplying Loss by $T^2$

To ensure gradient magnitudes remain stable regardless of temperature $T$, Geoffrey Hinton et al. scaled the soft cross-entropy loss function by $T^2$:

$$\mathcal{L}_{\text{soft}} = - T^2 \sum_i p_i \log(q_i)$$

When taking the derivative with respect to $z$, one factor of $T$ cancels out:

$$\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z} = T^2 \cdot \frac{1}{T}(q - p) = T(q - p)$$

This simple $T^2$ multiplier restores scale invariance, keeping gradient magnitudes steady across different choices of $T$.

### Empirical Proof in NumPy

```python
import numpy as np

def softmax_with_temperature(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    scaled_logits = (logits - np.max(logits, axis=-1, keepdims=True)) / T
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Teacher (v) and Student (z) logits
v = np.array([2.0, 5.0, 1.0])
z = np.array([1.0, 2.0, 0.0])

print(f"{'Temp (T)':<10} | {'Unscaled Grad (q - p)/T':<25} | {'Scaled Grad T * (q - p)':<25}")
print("-" * 65)

for T in [1.0, 5.0, 20.0, 100.0]:
    p = softmax_with_temperature(v, T)
    q = softmax_with_temperature(z, T)
    
    # Unscaled: Gradient shrinks by 1/T^2
    grad_unscaled = (q - p) / T
    
    # Scaled (T^2 factor): Gradient stays scale-invariant
    grad_scaled = T * (q - p)
    
    print(f"T = {T:<8} | {str(np.round(grad_unscaled, 5)):<25} | {str(np.round(grad_scaled, 4)):<25}")

## 4. High-Temperature Limit: Convergence to MSE

A fascinating mathematical property of Knowledge Distillation is what happens when temperature $T \to \infty$. At extremely high temperatures, **Soft Cross-Entropy optimization becomes mathematically equivalent to Mean Squared Error (MSE) on raw logits.**

### The Taylor Expansion Proof

Using the Taylor series approximation $\exp(x) \approx 1 + x$ for small $x = \frac{z}{T}$:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} \approx \frac{1 + z_i / T}{\sum_j (1 + z_j / T)} = \frac{1 + z_i / T}{N + \frac{1}{T}\sum_j z_j}$$

If we assume zero-mean centered logits ($\sum_j z_j = 0$ and $\sum_j v_j = 0$), this simplifies to:

$$q_i \approx \frac{1}{N}\left(1 + \frac{z_i}{T}\right) \quad \text{and} \quad p_i \approx \frac{1}{N}\left(1 + \frac{v_i}{T}\right)$$

Substituting these approximations into our scaled gradient formula $\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z_i} = T(q_i - p_i)$:

$$\frac{\partial \mathcal{L}_{\text{soft}}}{\partial z_i} \approx T \left( \frac{1}{N}\left(1 + \frac{z_i}{T}\right) - \frac{1}{N}\left(1 + \frac{v_i}{T}\right) \right) = \frac{1}{N}(z_i - v_i)$$

This is precisely the gradient of the **Mean Squared Error** loss $\mathcal{L}_{\text{MSE}} = \frac{1}{2N} \Vert{}z - v\Vert{}^2$ computed on zero-mean centered logits!

### Numerical Verification in NumPy

```python
import numpy as np

def softmax_with_temperature(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    scaled_logits = (logits - np.max(logits, axis=-1, keepdims=True)) / T
    exp_logits = np.exp(scaled_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

v = np.array([2.0, 5.0, 1.0])
z = np.array([1.0, 2.0, 0.0])

# 1. Zero-mean center the logits
v_centered = v - np.mean(v)
z_centered = z - np.mean(z)

# 2. Centered MSE Gradient: (z_centered - v_centered)
grad_mse_centered = z_centered - v_centered

# 3. Soft Cross-Entropy Gradient at High T (T = 50.0)
T = 50.0
p = softmax_with_temperature(v, T)
q = softmax_with_temperature(z, T)
grad_soft = T * (q - p)

# Multiply soft gradient by N=3 to place on the same scale
print(f"Centered MSE Gradient:           {np.round(grad_mse_centered, 4)}")
print(f"Soft Cross-Entropy Grad (T=50): {np.round(3 * grad_soft, 4)}")

**Output:**
```text
Centered MSE Gradient:           [ 0.6667 -1.3333  0.6667]
Soft Cross-Entropy Grad (T=50): [ 0.6644 -1.3289  0.6644]

At high temperatures, soft cross-entropy gradient matches zero-mean MSE gradient almost perfectly.
