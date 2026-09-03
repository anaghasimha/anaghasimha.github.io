---
layout: post
title: When "Poor" Means Everything — Construct Instability in NLP Classifiers
date: 2026-09-01
description: What happens when your classifier is confidently wrong at scale.
tags: [blog] 
---

Natural Language Processing (NLP) is a technology that has truly revolutionized how we process text data. Most of NLP's utilization has been in text summarization, recommendations, and filtering reviews, and it does a good job at all of these. But I believe that the biggest challenge for NLP is going to be social media, simply because most of the content is context-dependent. One post can be naive and the other can be harmful, and the difference lives entirely in context.

A simple example: imagine a video of a girl hilariously slipping while skating, and the comments say "poor girl." In this context, it means pity, not that she is financially poor. Now imagine a video about a girl in economic hardship, and the comments say the same thing. As a human, you distinguish these instantly. An NLP system, without context, often cannot.

This is a challenge I faced firsthand in my own research. I was studying economic grievance on extremist forums — online communities where users express resentment about economic conditions — using a large corpus of over 36 million posts spanning two decades. To label the data, I used Ollama, a large language model, to classify posts into three categories: no economic grievance, slight economic grievance, and high economic grievance. But the model was flagging almost any post containing the word "poor" as economic grievance, regardless of whether it meant financial hardship, poor quality, or simple pity. The labels were noisy, and I knew it.

This meant that DistilBERT, the classifier I fine-tuned on these labels, was being trained on noisy data. I didn't trust the output, so I ran more checks.

I tried training on DeBERTa, a larger and more capable model. Performance improved by only 0.018 F1 points. I suspected the problem might be data volume, so I ran a learning curve analysis. Both models showed similar performance regardless of how much data I used. I tried different prompts. Nothing moved the needle.

This ruled out the usual suspects: model capacity, data volume, prompt design. The problem was something deeper: the construct itself was unstable. "Economic grievance" means something different on a white supremacist forum than it does on a pickup artistry forum. The word "poor" carries different meanings across communities, and no single definition could capture all of them.

Now here is where it gets interesting. I had integrated my classifier outputs into a causal design — specifically a difference-in-differences framework — to estimate how macroeconomic surprises affect grievance posting across forums. The estimated effect was small but statistically significant. But should I trust it?

Think of it like watching TV while a fan is running. The fan's noise doesn't stop the TV from broadcasting. It just makes the signal harder to hear. You catch parts of it, but not clearly. This is what statisticians call attenuation bias: measurement error pulls the estimated effect toward zero, making a true effect look smaller than it actually is. My noisy classifier was the fan.

This is the part that keeps me up at night. When classifiers are deployed in social science research, their errors don't just affect prediction accuracy. They propagate into causal estimates, policy conclusions, and moderation decisions. A classifier that confidently labels wrong at scale, across millions of posts with no human in the loop, can silently distort what we think we know.

The solution isn't just better models. It's better measurement. NLP systems deployed on contested social constructs need to know when they don't know — to surface uncertainty rather than bury it under a confident wrong label. That's the problem I want to keep working on.
