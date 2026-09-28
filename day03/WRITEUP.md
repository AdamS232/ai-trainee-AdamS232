# Baseline vs. Transformer — TF-IDF+LogReg vs. DistilBERT

|                        | TF-IDF + LogReg (Day 2) | DistilBERT (Day 3) |
|------------------------|--------------------------|----------------------|
| Accuracy               | 89.0%                    | 90.1%                |
| F1 (macro)             | 0.8907                   | 0.9009               |
| Training time          | ~2 seconds (CPU)         | 2 min 20 sec (Colab T4 GPU, 1000 steps / 2 epochs) |
| Peak GPU mem           | —                        | ~2.2 GB (Colab T4, fp16, batch size 16) |
| Disk footprint (model) | 15 MB                    | ~255 MB (model.safetensors, fp32) |
| Inference latency (100 samples) | 0.06 s (~0.6 ms/review, CPU) | 9.9 s (~99 ms/review, CPU) |

The DistilBERT fine-tune beat the TF-IDF baseline on both accuracy (+1.1pp) and F1 (+0.01), despite training on a smaller 8k-example subset for only 2 epochs. The gain reflects DistilBERT's contextual understanding — it captures word order, negation, and sarcasm that a bag-of-words TF-IDF representation can't, since TF-IDF treats "not good" and "good" as just two independent tokens with no relationship.

That said, the cost difference is real: DistilBERT needs a GPU to train in reasonable time (it took under 2.5 minutes on a T4 vs. an estimated 2 hours on CPU), and its saved model is roughly ~17x larger on disk than the TF-IDF pipeline. Inference is also slower per request — a forward pass through 66M parameters vs. a sparse dot product.

**When I'd still pick TF-IDF + LogReg in production:** when inference must run on CPU-only infrastructure with tight latency SLAs (sub-millisecond per request), when the dataset is small (a few hundred to low-thousands of examples) and a transformer would overfit or offer no real lift, or when stakeholders need an interpretable model — TF-IDF+LogReg lets you inspect exactly which words drove a prediction via the model's coefficients, which DistilBERT cannot easily offer without extra tooling like SHAP or attention visualization.