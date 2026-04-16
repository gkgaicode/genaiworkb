# Get me effective prompt you can use directly with Claude or any AI coding assistant for : Task 6 — Real-World Questions

## 🗒️ Answer

Here's an effective prompt you can use directly with Claude or any AI coding assistant for Task 6 (Real-World Questions):

---

**Prompt:**

```
You are an expert NLP Engineer and Data Scientist with deep industry 
experience. I am building a Text Processing Pipeline project using 
product reviews and have implemented OHE, Bag of Words, and TF-IDF.

Now help me answer the following three real-world conceptual questions 
with detailed explanations, concrete examples, and Python code 
demonstrations where applicable:

1. WHY BAG OF WORDS FAILS AT SEMANTIC MEANING:
   - Explain the core limitation of BoW with word order
   - Demonstrate with Python code:
     * Show "The movie was fantastic" vs "The film was amazing"
       have zero word overlap but identical meaning
     * Show "not good" vs "good" appear similar in BoW
       but are opposite in meaning
   - Show BoW vector representations for these examples
   - Explain what is missing: synonyms, context, word order,
     negation handling
   - Mention what techniques solve this: Word2Vec, BERT, GloVe

2. WHEN TO USE BOW VS TF-IDF IN INDUSTRY:
   - Create a detailed comparison table:
     | Use Case | Recommended | Reason |
   - Cover these industry scenarios:
     * Spam detection
     * Search engine ranking
     * Document classification
     * Keyword extraction
     * Sentiment analysis
     * Recommendation systems
   - Explain with real company examples where possible
   - Provide Python code showing performance difference
     on a small classification task using both methods

3. LIMITATIONS OF TF-IDF IN REAL APPLICATIONS:
   - List and explain each limitation with a code example:
     * Ignores word order and grammar
     * No semantic understanding (synonyms treated as different)
     * Fixed vocabulary — fails on unseen words (OOV problem)
     * Poor performance on very short texts
     * IDF becomes noisy with small corpora
     * No handling of multi-word expressions (n-grams partially help)
   - For each limitation suggest a modern NLP alternative
   - Show a concrete failure case with actual review text samples
     from cleaned_reviews.csv

Additional requirements:
- Format each answer with clear section headers
- Include Python print statements for all explanations
  so output is self-documenting in the notebook
- Add a final summary table comparing BoW, TF-IDF, and
  modern alternatives (Word2Vec, BERT) across:
  * Semantic understanding
  * Speed
  * Memory usage
  * Industry adoption
  * Best use case
- List all required imports at the top
- Keep code examples concise but fully runnable
- Add inline comments explaining key concepts
```

---

This prompt is conceptually rich, example-driven, and industry-focused —
combining theory, code, and practical guidance following best practices
for structured AI coding prompts. [[1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)] [[2](https://sderosiaux.medium.com/how-i-learned-to-prompt-ai-better-my-four-modes-177bddcfa6bd)] [[5](https://pub.towardsai.net/the-prompts-i-actually-use-to-code-with-ai-assistants-a442597bd6c0)] [[6](https://aws.amazon.com/blogs/machine-learning/prompt-engineering-techniques-and-best-practices-learn-by-doing-with-anthropics-claude-3-on-amazon-bedrock/)]

## 🌐 Sources

1. [platform.claude.com - Prompting best practices - Claude API Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
2. [sderosiaux.medium.com - How I Learned to Prompt Claude Code Better — Four Modes](https://sderosiaux.medium.com/how-i-learned-to-prompt-ai-better-my-four-modes-177bddcfa6bd)
3. [reddit.com - The Only Prompt You Need : r/ClaudeAI](https://www.reddit.com/r/ClaudeAI/comments/1gds696/the_only_prompt_you_need/)
4. [github.com - langgptai/awesome-claude-prompts](https://github.com/langgptai/awesome-claude-prompts)
5. [pub.towardsai.net - The Prompts I Actually Use to Code with AI Assistants](https://pub.towardsai.net/the-prompts-i-actually-use-to-code-with-ai-assistants-a442597bd6c0)
6. [aws.amazon.com - Prompt engineering techniques and best practices](https://aws.amazon.com/blogs/machine-learning/prompt-engineering-techniques-and-best-practices-learn-by-doing-with-anthropics-claude-3-on-amazon-bedrock/)