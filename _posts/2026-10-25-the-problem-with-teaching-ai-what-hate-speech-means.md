---
layout: post
title: "The Problem with Teaching AI what Hate Speech Means"
date: 2026-10-25
tags: [blog]
---

Cambridge defines hate speech as “public speech that expresses hate or encourages violence towards a person or group based on something such as race, religion, sex, or sexual orientation”. But Wikipedia notes that there is no single, consistent definition for hate speech. When a concept has no fixed definition, how do we define its boundaries? And when we as humans cannot define a concept, how can we teach it to AI? These were the exact questions I faced in my research.

Going through 2000+ toxic posts, I felt that all posts were hateful or malicious. It was so difficult to not label posts with slurs as not hateful. Then, I saw posts by members of the LGBTQ+ community using the word “queer”. Initially, I thought “queer” was a slur, but later realized that it was a reclaimed word that LGBTQ members now used. This questioned my entire basis for labeling posts as toxic or not. If the presence or absence of words by certain members of the community can make so much difference, I wanted to be careful about who gets heard on the Internet. 

I was more careful now with my annotation strategies. I read multiple papers about hate speech and free speech to make sure I do not bury the voices that need to be heard the most. But I still failed at edge cases where I wasn’t sure if a post was hateful or not. Consider the example “These people are destroying our country.”. There are multiple ways of understanding this sentence - it can be political criticism, coded racism or just frustration. There is no way to understand the intention of the user through this sentence, even for a human being. An AI classifier is much further from understanding this context, which is what makes the task all the more nuanced. 

Despite being cautious about annotation, my classifier still got it wrong in many contexts. The issue didn’t stop here. These erroneous labels went to my downstream application and there were many wrong labels. How can I trust an application that has learned false signals? Should I be worried about my voice being trampled because an AI couldn’t get a definition right? There are many papers that have tried tackling this challenge and many have come to the same conclusion of construct instability of community-specific concepts. This is also what has motivated me to work on reference-free frameworks which do not require a fixed definition or information about user demographics to build a trustworthy NLP measurement tool. Uncertainty quantification methods can also help LLMs surface labels that were labeled doubtfully. Going through those posts changed how I think about what we’re building. Every wrong label is a decision about whose voice gets heard. And I want to build systems that get this decision right.

