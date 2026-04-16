# Get me effective prompt you can use directly with Claude or any AI coding assistant for : Task 4 — Comparison Analysis

## 🗒️ Answer

Here's an effective prompt you can use directly with Claude or any AI coding assistant for Task 4 (Comparison Analysis):

---

**Prompt:**

```
You are an expert NLP and Data Science engineer. I have already computed 
three feature matrices from cleaned_reviews.csv:
- ohe_matrix  → from CountVectorizer(binary=True)
- bow_matrix  → from CountVectorizer()
- tfidf_matrix → from TfidfVectorizer()

Write clean, modular Python code to perform a detailed comparison 
analysis of all three representations IN ORDER:

1. COMPARISON TABLE:
   - Create a pandas DataFrame comparing OHE, BoW, and TF-IDF
   - Select 5 sample reviews and 10 representative words
   - Show OHE values, BoW counts, and TF-IDF scores side by side
   - Print the table with clear column labels

2. TOP TF-IDF WORDS PER DOCUMENT:
   - For each of the first 5 reviews, extract top 5 TF-IDF words
   - Use argsort() on tfidf_matrix rows
   - Print: Review number → Top 5 words → Their TF-IDF scores

3. COMMON WORDS WEIGHT ANALYSIS:
   - Identify 5 most common words from vocabulary
   - Show their TF-IDF scores vs BoW counts
   - Explain numerically why common words receive lower TF-IDF weight

4. VISUAL COMPARISON:
   - Plot a grouped bar chart comparing BoW vs TF-IDF scores
     for top 15 words across 3 sample reviews
   - Use matplotlib with clear legends and axis labels

5. WRITTEN ANALYSIS (as Python print statements):
   - Why OHE loses frequency information
   - Why BoW ignores word importance across documents
   - Why TF-IDF is superior for discriminative word detection

Additional requirements:
- Add inline comments explaining each step
- List all required imports at the top
- Use clear section headers before each analysis block
- Handle sparse matrix conversion to dense where needed
  using .toarray()
```

---

This prompt is detailed, visually rich, and requests both quantitative and 
qualitative analysis — key traits of thorough comparison prompts for 
AI coding assistants. [[1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)] [[5](https://pub.towardsai.net/the-prompts-i-actually-use-to-code-with-ai-assistants-a442597bd6c0)] [[6](https://aws.amazon.com/blogs/machine-learning/prompt-engineering-techniques-and-best-practices-learn-by-doing-with-anthropics-claude-3-on-amazon-bedrock/)]

## 🌐 Sources

1. [platform.claude.com - Prompting best practices - Claude API Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
2. [sderosiaux.medium.com - How I Learned to Prompt Claude Code Better — Four Modes](https://sderosiaux.medium.com/how-i-learned-to-prompt-ai-better-my-four-modes-177bddcfa6bd)
3. [reddit.com - The Only Prompt You Need : r/ClaudeAI](https://www.reddit.com/r/ClaudeAI/comments/1gds696/the_only_prompt_you_need/)
4. [github.com - langgptai/awesome-claude-prompts](https://github.com/langgptai/awesome-claude-prompts)
5. [pub.towardsai.net - The Prompts I Actually Use to Code with AI Assistants](https://pub.towardsai.net/the-prompts-i-actually-use-to-code-with-ai-assistants-a442597bd6c0)
6. [aws.amazon.com - Prompt engineering techniques and best practices](https://aws.amazon.com/blogs/machine-learning/prompt-engineering-techniques-and-best-practices-learn-by-doing-with-anthropics-claude-3-on-amazon-bedrock/)