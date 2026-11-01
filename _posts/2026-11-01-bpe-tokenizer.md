---
layout: post
title: "How LLMs Read: Byte Pair Encoding"
date: 2026-11-01
tags: [notes]
---
Imagine having to learn every single word and punctuation mark from a book. Now you'll have to keep this in memory all your life. Later, you're given one more book and another that you have to keep in memory. That would be difficult right? Now imagine trying to teach an LLM all these words. Even with all the compute in the world, an LLM might learn every single word from every text document on the planet. But what would happen once it comes across an error? If modern LLMs broke down every time it saw an error, we wouldn't be where we are now. This is where subword tokenization finds the exact middle ground between memory efficiency and open-vocabulary expression.
Byte-pair encodings do not understand the meaning of the word. They instead focus on breaking the word into individual characters with the end-of-word marker (.). For example, if a paragraph contains the word lower, this is how a machine would break it down: l o w e r. Now it looks at its trained list of merge rules - these rules are stored in the exact chronological order in which they were learned). It checks if any pair in the word match against the highest-priority rule:

**Rule 1:** l + o → lo  
Result: lo w e r

**Rule 2:** lo + w → low  
Result: low e r .

**Rule 3:** e + r → er  
Result: er .

The word lower has now been successfully tokenized to `["low", "er"]`.

Even if the paragraph contains a new word, completely unseen by an LLM, it will use the same mechanics to break down the word. The tokenizer will not crash in this instance. Instead, it will keep chopping the words until the very last character to match the tokens in its vocabulary.

Earlier BPE focused only on alphabets. But what if a model encounters an emoji, non-English characters or code snippets with white spaces? The model would crash again. So, from GPT 2 onwards, companies started running BPE over raw bytes (0-255) over alphabets. This ensured that every single character could now be broken down into known tokens, eliminating the risk of unknown character errors.

BPE is a fundamental method used in all modern GenAI models. It offers real engineering trade-offs and better production realities. With vocabulary size as a hyperparameter, models can learn a small vocabulary which equates shorter training longer text sequences as opposed to a massive vocabulary that has shorter sequences and embedding matrices that can swallow GPU memory. Production tokenizers also require heavy optimization (like Rust implementations and pre-tokenization regex splits) so that preparing billions of tokens doesn't stall the training pipelines.
