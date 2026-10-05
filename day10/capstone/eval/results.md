# AskThePaper — Evaluation Results

17 questions over 3 PDFs (attention.pdf, my_document.pdf, sbert.pdf): 9 positive, 4 paraphrase, 4 out-of-scope.

## Summary

| Metric | Result |
|---|---|
| Retrieval hit rate (answerable) | 11/13 (85%) |
| Answer accuracy (answerable) | 8/13 (62%) |
|   - positive | 6/9 (67%) |
|   - paraphrase | 2/4 (50%) |
| Correct refusals (out-of-scope) | 4/4 (100%) |
| False refusals (answerable) | 1/13 (8%) |
| Inline [S#] citation rate (answered) | 11/12 (92%) |
| Time to first token, median / p95 | 4.83s / 4.95s |
| Total latency, median / p95 | 5.06s / 5.77s |

## Per-question results

| # | Type | Question | Retrieval hit | Correct | Cited | Time | Answer (truncated) |
|---|---|---|---|---|---|---|---|
| 1 | positive | How many attention heads does the base Transformer use? | ✅ | ✅ | ✅ | 10.1s | According to [S2], the Transformer employs 8 parallel attention layers, or heads. |
| 2 | paraphrase | In the Transformer paper, how many parallel attention layers run side by side? | ✅ | ✅ | ✅ | 4.9s | According to [S3], in this work, they employ h= 8 parallel attention layers, or heads, running side by side. |
| 3 | positive | How many identical layers are in the Transformer encoder stack? | ✅ | ✅ | ✅ | 5.1s | According to [S1], the Transformer encoder stack is composed of a stack of N= 6 identical layers. |
| 4 | positive | What is the model dimension d_model of the base Transformer? | ✅ | ❌ | ✅ | 5.1s | According to Table 3 in [S1], the model dimension d_model of the base Transformer is 512. |
| 5 | positive | What BLEU score did the big Transformer reach on WMT 2014 English-to-German? | ✅ | ❌ | — | 5.1s | I don't know based on the provided documents. |
| 6 | paraphrase | Which optimization algorithm was used to train the Transformer? | ❌ | ❌ | ✅ | 5.6s | According to [S2], the Transformer was trained using an optimization algorithm that is not explicitly mentione |
| 7 | positive | What two RAG formulations does the paper compare? | ✅ | ✅ | — | 5.2s | The paper compares two RAG formulations:   1. One which conditions on the same retrieved passages across the w |
| 8 | paraphrase | What component does RAG use to fetch relevant passages? | ✅ | ✅ | ✅ | 5.0s | According to [S3] (my_document.pdf, p.2), RAG uses a Dense Passage Retriever (DPR) to fetch relevant passages. |
| 9 | positive | Which pre-trained model does RAG use as its generator? | ✅ | ✅ | ✅ | 5.2s | According to [S2], the pre-trained seq2seq transformer (BART [32]) is used as the generator in RAG models. |
| 10 | positive | What knowledge source does RAG retrieve documents from? | ❌ | ❌ | ✅ | 5.1s | According to [S1], RAG retrieves documents from its non-parametric memory. |
| 11 | positive | What network structures does Sentence-BERT use to fine-tune BERT? | ✅ | ✅ | ✅ | 5.8s | According to [S1] (sbert.pdf, p.3), Sentence-BERT uses two network structures:   1. Siamese network structure: |
| 12 | paraphrase | How much faster is SBERT than BERT at finding the most similar pair among 10,000 sentences? | ✅ | ❌ | ✅ | 5.3s | According to [S1], SBERT is about 9% faster than BERT at finding the most similar pair among 10,000 sentences. |
| 13 | positive | Which pooling strategy does SBERT use by default? | ✅ | ✅ | ✅ | 5.0s | The default pooling strategy used by SBERT is MEAN. [S1] |
| 14 | out_of_scope | What is the population of Canada? | — | ✅ | — | 2.3s | I don't know based on the provided documents. |
| 15 | out_of_scope | Who won the 2018 FIFA World Cup? | — | ✅ | — | 2.4s | I don't know based on the provided documents. |
| 16 | out_of_scope | What is a good recipe for chocolate chip cookies? | — | ✅ | — | 2.4s | I don't know based on the provided documents. |
| 17 | out_of_scope | How many GPUs were used to train GPT-4? | — | ✅ | — | 5.0s | I don't know based on the provided documents. |

## Method

- **Retrieval hit:** a returned source is from the expected PDF *and* contains an expected keyword.
- **Correct (answerable):** the answer is not a refusal and contains an expected keyword.
- **Correct (out-of-scope):** the answer is the refusal "I don't know based on the provided documents."
- Timed end-to-end through the API (`/chat`, streamed), on an RTX 4070 laptop with llama3.2:3b in Ollama.
- Keyword matching is a rough proxy: it can't judge wording quality and could miss a correct answer phrased differently.