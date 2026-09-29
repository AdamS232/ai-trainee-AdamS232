# Day 6 — Elasticsearch Queries over IMDB

Index: `imdb_reviews` (25,000 training reviews). Fields: `text` (text, english analyzer), `sentiment` (keyword), `length` (integer).

## 1. "best cinematography" with sentiment = positive

```json
{ "bool": {
    "must":   [{ "match": { "text": "best cinematography" } }],
    "filter": [{ "term":  { "sentiment": "positive" } }] } }
```

**What it does:** `match` runs the words through the english analyzer and scores reviews with BM25. `filter` keeps only positive reviews without changing the scores (like SQL `WHERE`).

**Result:** All 5 results were positive (the filter works), and all 5 contain both words. Short reviews that mention both rank high because of BM25 length normalization. Adding `"operator": "and"` (require both words) gave the same top 5 here, since the best matches already had both.

## 2. Distribution of review lengths across sentiments

```json
{ "size": 0,
  "aggs": { "by_sentiment": {
    "terms": { "field": "sentiment" },
    "aggs": {
      "length_stats":   { "stats": { "field": "length" } },
      "length_buckets": { "histogram": { "field": "length", "interval": 1000 } } } } } }
```

**What it does:** Groups reviews by sentiment (like SQL `GROUP BY`), then computes length stats and a histogram for each group. `size: 0` returns only the numbers, not the reviews.

**Result:**

| Sentiment | Reviews | Avg length | Min | Max |
|---|---|---|---|---|
| Negative | 12,500 | 1,303 chars | 52 | 8,969 |
| Positive | 12,500 | 1,347 chars | 70 | 13,704 |

The classes are perfectly balanced and the length distributions are nearly identical: about half of all reviews are under 1,000 characters. Positive reviews are slightly more common at 3,000+ characters. Length alone can't predict sentiment.

## 3. multi_match with "brilliant" boosted 2x

```json
{ "bool": {
    "must":   [{ "multi_match": { "query": "brilliant acting", "fields": ["text"] } }],
    "should": [{ "match": { "text": { "query": "brilliant", "boost": 2 } } }] } }
```

**What it does:** `multi_match` searches the query across a list of fields. The `should` clause is optional, but when "brilliant" appears its score counts double.

**Result:** Top scores were about 17–18, more than twice Query 1's scores, because "brilliant" is counted in both clauses. Reviews using "brilliant" rank above ones that only mention "acting." Scores can't be compared between different queries, only the ranking within one query.

## 4. Phrase query: "waste of time"

```json
{ "match_phrase": { "text": "waste of time" } }
```

**What it does:** Requires the words to appear together, in order, unlike `match`, which accepts them anywhere in the review.

**Result:** 787 matches, 739 of them negative (94%), so it's a strong negative signal. The top hit said "waste *your* time": the english analyzer drops the stopword "of" but keeps its position, so any word in that slot matches. The 48 positive matches likely negate the phrase ("*not* a waste of time"), which phrase search can't detect.

## 5. more_like_this: reviews similar to review #0

```json
{ "more_like_this": {
    "fields": ["text"],
    "like": [{ "_index": "imdb_reviews", "_id": "0" }],
    "min_term_freq": 1,
    "max_query_terms": 25 } }
```

**What it does:** Picks the 25 most distinctive words from the source review (by TF-IDF) and finds reviews that share them.

**Result:** The source review was about *I Am Curious (Yellow)*, and all 5 results were about the same film. But their opinions differed ("a masterwork" vs. "this ridiculous film"): more_like_this matches the topic through word overlap, not the opinion.