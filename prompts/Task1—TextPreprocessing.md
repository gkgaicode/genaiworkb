# Get me good prompt to code : Task 1 — Text Preprocessing

## 🗒️ Answer

Here's an effective prompt you can use directly with Claude or any AI coding assistant for Task 1 (Text Preprocessing):

---

**Prompt:**

```
You are an expert NLP engineer. I have a CSV file called reviews.csv with a 
column named review_text containing real product reviews scraped from Flipkart.

Write a clean, modular Python function called preprocess(text) that performs 
the following steps IN ORDER:

1. Convert text to lowercase
2. Remove URLs, HTML tags, and special characters using regex
3. Tokenize the text using nltk.word_tokenize()
4. Remove punctuation using string.punctuation
5. Remove English stopwords using nltk.corpus.stopwords
6. Apply lemmatization using nltk.stem.WordNetLemmatizer

Additional requirements:
- Apply preprocess() to all rows in reviews.csv
- Store cleaned text in a new column: clean_text
- Save updated dataframe to cleaned_reviews.csv
- Print before/after example for 3 sample reviews
- Handle NaN or empty strings gracefully
- Add inline comments explaining each step
- List all required library imports at the top

Show sample output after preprocessing is complete.
```

---

This prompt is precise, ordered, and requests modular and well-commented code — key traits of effective AI coding prompts. [[5](https://www.altersquare.io/5-ai-prompts-every-developer-should-master-copy-paste-ready/)] [[6](https://towardsdatascience.com/advanced-prompt-engineering-for-data-science-projects/)]

## 🌐 Sources

1. [reddit.com - I compiled 200 advanced Claude prompts for coding](https://www.reddit.com/r/PromptEngineering/comments/1sfcosw/i_compiled_200_advanced_claude_prompts_for_coding/)
2. [medium.com - Ultimate Prompts for Every Developer | by Onix React](https://medium.com/@onix_react/ultimate-prompts-for-every-developer-031a6d26a569)
3. [community.openai.com - Sourcing useful ChatGPT coding prompts to feature - Codex](https://community.openai.com/t/sourcing-useful-chatgpt-coding-prompts-to-feature/1357452)
4. [dataplatform.cloud.ibm.com - Sample foundation model prompts for common tasks](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-prompt-samples.html?context=wx)
5. [altersquare.io - 5 AI Prompts Every Developer Should Master](https://www.altersquare.io/5-ai-prompts-every-developer-should-master-copy-paste-ready/)
6. [towardsdatascience.com - Advanced Prompt Engineering for Data Science Projects](https://towardsdatascience.com/advanced-prompt-engineering-for-data-science-projects/)