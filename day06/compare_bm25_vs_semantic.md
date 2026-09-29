# BM25 (Elasticsearch) vs Semantic Search (FAISS)

Both engines searched the **same 20,000 chunks** of Simple English Wikipedia from Day 5, with the same 10 test queries. BM25 used the english analyzer; semantic search used `all-MiniLM-L6-v2` embeddings.

| Engine | Recall@5 |
|---|---|
| BM25 | **9/10** |
| Semantic | 8/10 |

## Example 1: BM25 wins on specific keywords (Alan Turing)

**Query:** "British mathematician who broke German codes in World War 2"

- **BM25:** HIT (#2): `May 9, Alan Turing, Luftwaffe, 1943, October 12`
- **Semantic:** MISS: `December 18, List of mathematicians, December 8, November 20, April 9`

The Alan Turing chunk contains rare words like "mathematician," "German," and "codes," and BM25 rewards rare exact matches (high IDF). The embedding model instead matched the *general feel* of the query ("British person + job + war") to date pages, which are long lists like "1912 – X, British mathematician." Semantic search struggles when the answer depends on specific facts rather than overall topic.

## Example 2: semantic ranks better on paraphrases (Farming)

**Query:** "growing crops and raising animals for food"

- **BM25:** HIT, but only at #4: `Farm, Farmer, Tropical rainforest, Farming, Afghanistan`
- **Semantic:** HIT at #1, and 4 of the top 5 are the Farming article: `Farming, Farm, Farming, Farming, Farming`

Both technically "hit," but semantic search understood that the query *describes* farming, even though the word "farming" never appears. BM25 only found it by matching "crops," "animals," and "food," and it also returned noise like "Tropical rainforest" and "Afghanistan." Recall@5 counts both as hits, but semantic's ranking is clearly better.

## Example 3: both fail when there are several right answers (Arithmetic)

**Query:** "adding, subtracting, multiplying and dividing numbers"

- **BM25:** MISS: `0, Algebra, Algebra, Algebra, Matrix (mathematics)`
- **Semantic:** MISS: `Algebra, Addition, Subtraction, Multiplication, Addition`

Neither engine found "Arithmetic," but semantic's results are more useful: each operation in the query has its own article, and it returned them. BM25's results (like "0" and "Matrix") are less relevant. The test was too strict, since several articles are reasonable answers.

## Other observations

- **Rare exact names help both engines.** "Jagged Little Pill" gave both engines Alanis Morissette in the top 3.
- **Both engines handled the common-knowledge paraphrases**: Air, Art, April, and August.
- **Why BM25 did better than expected:** each chunk starts with its article title (for example, "Farming: ..."), which gives BM25 an exact keyword to match. My queries also contained many specific words (Augustus, Louisiana, vector graphics). A test with pure paraphrases that share no words with the article would likely favor semantic search.

## Conclusion

BM25 is strong when the query contains **specific, rare words** (names, technical terms, facts). Semantic search is strong at **understanding what a query describes** and ranking the best article first, but it can be distracted by text with a similar feel, like date lists. In practice, many systems use **hybrid search**, combining both scores, to get the strengths of each.