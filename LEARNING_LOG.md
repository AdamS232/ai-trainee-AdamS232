# Learning Log

## Day 1 — Self-reflection answers
1. The difference between pip and mamba is that pip installs Python packages only, while mamba manages whole environments. People tend to use mamba for environment/system level stuff, while pip for Python packages ontop.

2. Pinning makes sure all installs are reproducible, so it makes sure that everyone gets the same versions. Its important because if the library updates later it doesn't silently break or change.

3. It would return False because without "--gpus all", the container can't see the host's GPU at all.

## Day 3 — DistilBERT fine-tune
- W&B run: https://wandb.ai/adam234-datum-analysis/ai-trainee/runs/b5paat8i
- Result: 90.1% accuracy / 0.9009 F1 (vs. 89.0% / 0.8907 for the Day 2 TF-IDF baseline)
- Trained on a Colab T4 GPU in 2 min 20 sec because my laptop has no GPU
- Write-up: day03/WRITEUP.md