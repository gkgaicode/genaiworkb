# Get me effective prompt you can use directly with Claude or any AI coding assistant for : Task 5 — Sparse Matrix Analysis

## 🗒️ Answer

Here's an effective prompt you can use directly with Claude or any AI coding assistant for Task 5 (Sparse Matrix Analysis):

---

**Prompt:**

```
You are an expert NLP and Data Science engineer. I have already computed 
three feature matrices from cleaned_reviews.csv:
- ohe_matrix  → from CountVectorizer(binary=True)
- bow_matrix  → from CountVectorizer()
- tfidf_matrix → from TfidfVectorizer()

Write clean, modular Python code to perform a thorough sparse matrix 
analysis of all three representations IN ORDER:

1. MATRIX SHAPE ANALYSIS:
   - Print .shape of each matrix (documents × vocab size)
   - Display in a formatted summary table:
     | Matrix | Rows | Columns | Total Elements |
   - Explain what rows and columns represent in each matrix

2. SPARSITY CALCULATION:
   - Calculate sparsity for each matrix using formula:
     sparsity = 1 - (matrix.nnz / (rows * cols))
   - Express as percentage rounded to 2 decimal places
   - Print results in formatted table:
     | Matrix | Non-Zero Elements | Total Elements | Sparsity % |
   - Identify which matrix is most/least sparse

3. MEMORY USAGE COMPARISON:
   - Calculate memory used by sparse format (CSR) in KB and MB
   - Calculate memory that would be used by equivalent dense matrix
   - Print memory savings achieved by sparse storage
   - Use sys.getsizeof() and matrix.data.nbytes for accurate measurement

4. SCALE SIMULATION:
   - Simulate what happens at 10K, 100K, 1M documents
   - Project matrix dimensions and memory requirements at each scale
   - Print a scaling table showing growth in size and memory
   - Show at what scale dense matrices become completely infeasible

5. VISUAL ANALYSIS:
   - Plot sparsity pattern of first 50 rows using matplotlib spy()
     for all three matrices side by side
   - Plot a bar chart comparing sparsity % across OHE, BoW, TF-IDF
   - Plot memory comparison: sparse vs dense for all three matrices

6. WRITTEN EXPLANATION (as Python print statements):
   - Why sparse matrices occur in NLP feature engineering
   - Why dense storage is inefficient for large-scale NLP systems
   - How CSR (Compressed Sparse Row) format solves memory problems
   - When to consider dimensionality reduction (PCA, SVD) instead

Additional requirements:
- Convert to dense only when needed using .toarray()
- Use scipy.sparse to inspect matrix properties
- Add inline comments explaining each calculation
- List all required imports at the top (numpy, scipy, sys, matplotlib)
- Use clear section headers before each analysis block
- Handle potential MemoryError gracefully with try/except
  when simulating large scale matrices
```

---

This prompt is thorough, quantitative, and visually comprehensive — 
covering memory, scale, and storage analysis with best practices for 
structured AI coding prompts. [[1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)] [[5](https://pub.towardsai.net/the-prompts-i-actually-use-to-code-with-ai-assistants-a442597bd6c0)] [[6](https://aws.amazon.com/blogs/machine-learning/prompt-engineering-techniques-and-best-practices-learn-by-doing-with-anthropics-claude-3-on-amazon-bedrock/)]

## 🌐 Sources

1. [platform.claude.com - Prompting best practices - Claude API Docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
2. [sderosiaux.medium.com - How I Learned to Prompt Claude Code Better — Four Modes](https://sderosiaux.medium.com/how-i-learned-to-prompt-ai-better-my-four-modes-177bddcfa6bd)
3. [reddit.com - The Only Prompt You Need : r/ClaudeAI](https://www.reddit.com/r/ClaudeAI/comments/1gds696/the_only_prompt_you_need/)
4. [github.com - langgptai/awesome-claude-prompts](https://github.com/langgptai/awesome-claude-prompts)
5. [pub.towardsai.net - The Prompts I Actually Use to Code with AI Assistants](https://pub.towardsai.net/the-prompts-i-actually-use-to-code-with-ai-assistants-a442597bd6c0)
6. [aws.amazon.com - Prompt engineering techniques and best practices](https://aws.amazon.com/blogs/machine-learning/prompt-engineering-techniques-and-best-practices-learn-by-doing-with-anthropics-claude-3-on-amazon-bedrock/)