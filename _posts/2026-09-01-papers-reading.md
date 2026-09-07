---
layout: post
title: Papers I'm Reading
date: 2026-09-01
description: A running log of papers I'm reading and what I'm taking away from them.
tags: [reading]
---

## August 2026

**Röttger et al. (2022) — "Two Contrasting Data Annotation Paradigms for Subjective NLP Tasks"**
Descriptive annotation captures how people actually use language; prescriptive annotation enforces a fixed definition. Neither paradigm addresses construct instability across communities — that's the gap my work sits in.

---

**Kozlowski, Taddy & Evans (2019) — "The Geometry of Culture"**
Word embeddings capture cultural meaning as geometric relationships. Gender and class associations shift diachronically (across time). My work shows the same instability synchronically (across communities at the same time). Motivates community-specific word embeddings as a reference-free evaluation approach.

---

**Horta Ribeiro et al. (2023) — "Deplatforming did not decrease Parler users' activity on fringe social media"**
Deplatforming is ineffective at the aggregate level — fringe activity increased after Parler was taken down. My work adds the individual-level mechanism: casual users disappear, the most extreme users migrate. Together: the aggregate finding is explained by who migrates.

---

**Sap et al. (2022) — "Annotators with Attitudes"**
Annotator demographics and beliefs systematically bias toxicity ratings. Same instability at the annotator level as I found at the community level. Single annotator limitation in my work is more serious than I initially acknowledged.

---

**de Kock (2024) — "Jointly modelling the evolution of social structure and language in online extremist groups"**
Dynamic user embeddings and K-means clustering find sub-communities that co-evolve in language and structure. Future work: bot detection via temporal user embeddings — my 10-lifetime-post threshold doesn't filter bots.

---

**Li et al. (2026) — "The Proxy Presumption: From Semantic Embeddings to Valid Social Measures"**

Geometric distance in embedding space is an unreliable proxy for social constructs without explicit validation. Embeddings conflate the target construct with confounds like topic and style. Their Construct Validity Protocol addresses this. My work provides empirical evidence of this problem: the word "poor" in embedding space conflates financial hardship with unrelated uses, producing systematic misclassification.

---

**Plaza-del-Arco et al. — "Respectful or Toxic? Using Zero-Shot Learning with Language Models to Detect Hate Speech"**

Zero-shot learning with prompting is comparable to fine-tuned models, especially for under-resourced languages. Prompt selection matters a lot, different prompts give very different results. They don't address construct instability and assume hate speech has a stable definition across contexts. Possible future direction: try zero-shot learning with prompting.

---

**Nefriana et al. — "Leader-driven or Leaderless: How Participation Structure Sustains Engagement and Shapes Narratives in Online Hate Communities"**

Differences in groups and leaders of antisemitic and Islamophobic content. In antisemitic communities, the agenda is more diverse; in Islamophobic communities, topics are narrower. Spike in activity during the Israel-Hamas war. Connects to my finding that toxicity is carried by a small group of people. Policymakers need to know the effects of blanket bans, as this doesn't reduce hate speech propagation.

---

**Umansky et al. (2026) — "Improving Hate Speech Detection with Large Language Models"**

Better annotations lead to better model performance, trained RAs outperformed crowd workers and citizen scientists. LLMs lack context and domain-specificity. Fine-tuning with high-quality domain-specific annotations is crucial. Parallel finding: human-labeled posts (F1=0.59) outperformed LLM-labeled expansion (F1=0.55). Cite for IAA experiment: shows that without deep understanding of the work, annotation is unreliable.

---

**Ross et al. (2016) — "Measuring the Reliability of Hate Speech Annotations: The Case of the European Refugee Crisis"**

Giving annotators a definition didn't improve agreement. Annotators relied on personal feelings rather than the definition. Inter-annotator agreement was low overall (Krippendorff's alpha 0.19-0.38). Two connections to my work: confirms hate speech is inherently unstable as a construct (same as economic grievance); shows why crowd workers fail (consistent with Donnay et al.). My two-annotator design is stronger than what Ross tested. Pagel and I are domain experts who have read thousands of these posts, not random crowd workers.

---

**Hosseinmardi et al. (2021) — "Examining the Consumption of Radical Content on YouTube"**

Far-right content consumers on YouTube are a small and stable percentage. Connects to my finding that very few users migrated to ExtremeBB after the Reddit ban, but those who did were more toxic. Both findings suggest radicalization is not primarily platform-driven but individual-driven, which has implications for how we design moderation interventions. Platform moderators should concentrate on the core small group propagating harmful content rather than banning entire accounts.


*Updated regularly. These are my honest takeaways, not summaries.*
