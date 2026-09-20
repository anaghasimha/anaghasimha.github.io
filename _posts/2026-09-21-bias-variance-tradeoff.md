---
layout: post
title: "Bias-Variance Tradeoff"
date: 2026-09-21
description: A beginner-friendly introduction to bias-variance tradeoff in machine learning.
tags: ml explainer
---
In today’s post, I’ll be discussing bias-variance trade-off, lesser in technical depth, but more towards simple understanding about the topic at hand. As and when I write more blogs, I intend to make them more technical.
Bias in machine learning is the inability of a machine learning method to capture the true relationship between variables and target.

<img width="468" height="126" alt="image" src="https://github.com/user-attachments/assets/lb-hb.jpeg" />

The straight line in the first image shows high bias i.e., the error between actual and predicted values on the testing set is high. Whereas in the squiggly line in the second image, the same error is very low, meaning low bias.

Variance is the difference in fits between different data sets. 

<img width="468" height="84" alt="image" src="https://github.com/user-attachments/assets/lv-hv.jpeg" />

The straight line in the first image shows low variance i.e., the error between actual and predicted values on the testing set is comparatively low. Whereas in the squiggly line in the second image, the same error in very high, meaning high variance.
Ideally, we would prefer a machine learning model that has low bias and low variance, but practical models don’t behave that way. There are several methods like regularization, bagging and boosting to obtain the ‘sweet spot’ between the two extremes of bias and variance.
Because the squiggly line fits the training set perfectly, but performs poorly on the testing set, we say that the second image is overfit. 
When the model doesn’t fit the testing set well, then the model is said to be underfitting.
We can use the formula below to check if the model is overfitting (high variance) or if we need to use more complex models: 
Err(x₀) = E[(Y − f̂(x₀))² | X = x₀] = σ²ε (Irreducible error) +[E f̂(x₀) − f(x₀)]² (Bias2) + E[f̂(x₀) − E f̂(x₀)]² (Variance)
Err(x₀) = E[(Y − f̂(x₀))² | X = x₀] = expected prediction error at point x₀ i.e., how wrong is our model on average at this point.
σ²ε (Irreducible error) = noise in the data itself, no matter how good the model is, more like randomness in the real world.
[E f̂(x₀) − f(x₀)]² (Bias2) = how far our average prediction is from truth.
E[f̂(x₀) − E f̂(x₀)]² (Variance) = how much do our predictions fluctuate across different training sets. 
References: 
1. StatQuest (2018). Bias and Variance. YouTube. https://youtu.be/EuBBz3bI-aA?si=bOGTF8ejnq86YWsM
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. 


*On AI use: This content has been written entirely by me. This includes ideation, writing and fact-checking. I haven't used AI to polish the sentences/check for spelling or grammar mistakes. The images in this post have been generated with the help of Gemini.*



