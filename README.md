# 🧠 Sentiment Analysis — VADER vs RoBERTa on Amazon Reviews

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/RoBERTa-0052CC?style=flat-square"/>
  <img src="https://img.shields.io/badge/VADER-00A86B?style=flat-square"/>
  <img src="https://img.shields.io/badge/NLTK-3776AB?style=flat-square"/>
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white"/>
</p>

A **side-by-side comparison of two fundamentally different NLP approaches** to sentiment analysis — classic lexicon-based VADER vs. transformer-based RoBERTa — applied to 500 Amazon Fine Food Reviews. Shows exactly where each model succeeds and fails, and why context matters.

---

## The Core Question This Project Answers

> Does a transformer that understands word context actually outperform a bag-of-words lexicon approach on real-world review data?

To answer this, both models are run on the same 500 reviews and their scores are compared directly via a pairplot — you can see the divergence clearly.

---

## Dataset

**Amazon Fine Food Reviews** (Kaggle)  
- Source: `/kaggle/input/amazon-fine-food-reviews/Reviews.csv`  
- Full dataset: 568,454 reviews  
- Used: first **500 reviews** (`df.head(500)`)  
- Star ratings: 1–5 (used as ground truth label)

---

## Two Models, Two Philosophies

### Model 1 — VADER (Valence Aware Dictionary and Sentiment Reasoner)

```python
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sia.polarity_scores("I am so sad")
# → {'neg': 0.6, 'neu': 0.4, 'pos': 0.0, 'compound': -0.5423}
```

**How it works:**
- Bag-of-words approach — each word is scored independently from a pre-built sentiment lexicon
- Stop words (`and`, `the`) are removed
- Individual word scores are combined into a final `compound` score (-1 to +1)
- No understanding of context, negation across words, or sarcasm

**Output scores:** `neg`, `neu`, `pos`, `compound`

### Model 2 — RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
```

**How it works:**
- Full transformer encoder — attends to every word in context of every other word
- Pre-trained on a large Twitter corpus, fine-tuned for 3-class sentiment
- Understands negation ("not good"), emphasis ("absolutely terrible"), and relative sentiment
- `softmax` converts raw logits to probabilities that sum to 1.0

**Output scores:** `roberta_neg`, `roberta_neu`, `roberta_pos`

---

## Pipeline

```
Reviews.csv (568k rows)
        │
        ▼
df.head(500)              ← first 500 reviews
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
VADER scoring                          RoBERTa scoring
SentimentIntensityAnalyzer             cardiffnlp/twitter-roberta-base-sentiment
  → vader_neg                            → roberta_neg
  → vader_neu                            → roberta_neu
  → vader_pos                            → roberta_pos
  → vader_compound
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
              results_df (merged with original df)
              columns: vader_*, roberta_*, Score, Text, Id
                       │
                       ▼
              sns.pairplot(hue='Score')
              → visual comparison across all 6 score dimensions
```

---

## EDA — Exploratory Data Analysis

Before modelling, the notebook explores the distribution of star ratings:

```python
df["Score"].value_counts().sort_index().plot(
    kind="bar",
    title="Count of reviews by stars",
    figsize=(10, 5)
)
```

And NLP preprocessing with NLTK:
```python
tokens  = nltk.word_tokenize(example)    # tokenise
tagged  = nltk.pos_tag(tokens)           # part-of-speech tagging
entities = nltk.chunk.ne_chunk(tagged)   # named entity recognition
entities.pprint()
```

This confirms the text structure before applying sentiment models.

---

## VADER — Running on Full 500 Reviews

```python
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
```

Compound score vs. star rating:
```python
sns.barplot(data=vaders, x='Score', y='compound')
# Expectation: higher stars → higher compound score
```

Positive / Neutral / Negative breakdown by star:
```python
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
```

---

## Head-to-Head Comparison

Both models run on every review, combined into a single results DataFrame:

```python
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text  = row["Text"]
        myid  = row["Id"]
        vader_result  = {f"vader_{k}": v for k, v in sia.polarity_scores(text).items()}
        roberta_result = polarity_scores_roberta(text)
        res[myid] = {**vader_result, **roberta_result}
    except RuntimeError:
        print(f"Broke for id {myid}")   # handles tokenizer length overflow
```

Final pairplot — all 6 scores against each other, coloured by star rating:

```python
sns.pairplot(
    data=results_df,
    vars=['vader_neg', 'vader_neu', 'vader_pos',
          'roberta_neg', 'roberta_neu', 'roberta_pos'],
    hue='Score',
    palette='tab10'
)
```

This single plot reveals where the two models agree, where they diverge, and which is better correlated with the true star rating.

---

## Key Findings

- **VADER** is fast and works well for clearly positive/negative reviews but struggles with nuanced, context-dependent sentiment
- **RoBERTa** captures context — a phrase like *"not bad at all"* scores differently between VADER (penalises "bad") and RoBERTa (understands the negation)
- **Agreement is high** on extreme ratings (1-star, 5-star); **divergence is highest** on 3-star reviews where sentiment is genuinely mixed

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Dataset | Amazon Fine Food Reviews (Kaggle) |
| Lexicon model | VADER (`nltk.sentiment.SentimentIntensityAnalyzer`) |
| Transformer model | `cardiffnlp/twitter-roberta-base-sentiment` (Hugging Face) |
| NLP preprocessing | NLTK (tokenisation, POS tagging, NER) |
| Inference | PyTorch + `softmax` (scipy) |
| Visualisation | Matplotlib, Seaborn (`barplot`, `pairplot`) |
| Data handling | Pandas, NumPy |
| Environment | Kaggle Notebooks |

---

## Setup

```bash
git clone https://github.com/utkarsh027/Sentiment_Analysis
cd Sentiment_Analysis
pip install transformers torch nltk pandas numpy matplotlib seaborn tqdm scipy
```

**Download NLTK data:**
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
```

**Dataset:** Download `Reviews.csv` from [Kaggle — Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and update the path in the notebook.

---

## Project Structure

```
Sentiment_Analysis/
├── sentiment-analysis.ipynb    # Full notebook: EDA → VADER → RoBERTa → comparison
└── README.md
```

---

## Author

**Utkarsh Tiwari** — MSc Data Science, UC Irvine  
[github.com/utkarsh027](https://github.com/utkarsh027) · [LinkedIn](https://www.linkedin.com/in/utkarsh-tiwari-330857166/)
