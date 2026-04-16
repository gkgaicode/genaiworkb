# Get me good prompt to code : Phase 0 — Data collection

## 🗒️ Answer

Here's an effective prompt you can use directly with Claude or any AI coding assistant for Phase 0 (Data Collection):

---

**Prompt:**

```
You are an expert Python web scraper. Help me scrape product reviews from 
Flipkart (or Amazon) for a text processing pipeline project.

Requirements:
- Scrape minimum 100 reviews from a high-review product (e.g. earphones)
- Use BeautifulSoup + requests (or Selenium if JS-rendered)
- Handle pagination to collect enough reviews
- Extract: review_text, star_rating, review_title
- Add polite delays (time.sleep) between requests
- Spoof User-Agent headers to avoid blocks
- Save output to reviews.csv with columns: review_text, rating, review_title
- Handle exceptions gracefully (timeouts, missing fields)
- Print progress every 10 reviews scraped

After writing the code, show a sample of the first 5 rows of the CSV output.
Do not scrape restricted or sensitive content.
```

---

This prompt follows key best practices: it is specific, defines output format, handles edge cases, and requests modular code. [[5](https://www.altersquare.io/5-ai-prompts-every-developer-should-master-copy-paste-ready/)] [[6](https://towardsdatascience.com/advanced-prompt-engineering-for-data-science-projects/)]

## 🌐 Sources

1. [reddit.com - I compiled 200 advanced Claude prompts for coding](https://www.reddit.com/r/PromptEngineering/comments/1sfcosw/i_compiled_200_advanced_claude_prompts_for_coding/)
2. [medium.com - Ultimate Prompts for Every Developer | by Onix React](https://medium.com/@onix_react/ultimate-prompts-for-every-developer-031a6d26a569)
3. [community.openai.com - Sourcing useful ChatGPT coding prompts to feature - Codex](https://community.openai.com/t/sourcing-useful-chatgpt-coding-prompts-to-feature/1357452)
4. [dataplatform.cloud.ibm.com - Sample foundation model prompts for common tasks](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-prompt-samples.html?context=wx)
5. [altersquare.io - 5 AI Prompts Every Developer Should Master (Copy-Paste Ready)](https://www.altersquare.io/5-ai-prompts-every-developer-should-master-copy-paste-ready/)
6. [towardsdatascience.com - Advanced Prompt Engineering for Data Science Projects](https://towardsdatascience.com/advanced-prompt-engineering-for-data-science-projects/)