| Embedder | Chunker | Size (words) | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR | AUROC | Missed |
|---|---|---|---|---|---|---|---|---|---|
| MiniLM | section | 120 | 623 | 76% | 96% | 96% | 0.85 | 0.86 | uc_t11 |
| MiniLM | section | 220 | 341 | 76% | 96% | 100% | 0.84 | 0.91 | — |
| MiniLM | section | 400 | 217 | 68% | 88% | 96% | 0.79 | 0.83 | uc_t11 |
| MiniLM | fixed | 120 | 631 | 72% | 84% | 92% | 0.79 | 0.87 | uc_t11,uc_t21 |
| MiniLM | fixed | 220 | 281 | 72% | 80% | 92% | 0.79 | 0.89 | uc_t21,uc_t24 |
| MiniLM | fixed | 400 | 141 | 60% | 80% | 84% | 0.68 | 0.85 | uc_t21,uc_t22,uc_t24,uc_t25 |
| BGE-small | section | 120 | 623 | 76% | 100% | 100% | 0.88 | 0.93 | — |
| BGE-small | section | 220 | 341 | 80% | 100% | 100% | 0.90 | 0.88 | — |
| BGE-small | section | 400 | 217 | 80% | 100% | 100% | 0.90 | 0.87 | — |
| BGE-small | fixed | 120 | 631 | 80% | 96% | 96% | 0.87 | 0.90 | uc_t21 |
| BGE-small | fixed | 220 | 281 | 64% | 84% | 92% | 0.74 | 0.86 | uc_t01,uc_t21 |
| BGE-small | fixed | 400 | 141 | 60% | 80% | 84% | 0.69 | 0.81 | uc_t01,uc_t21,uc_t24,uc_t25 |
| TF-IDF | section | 120 | 623 | 64% | 84% | 88% | 0.74 | 0.64 | uc_t12,uc_t17,uc_t23 |
| TF-IDF | section | 220 | 341 | 72% | 88% | 92% | 0.80 | 0.77 | uc_t17,uc_t23 |
| TF-IDF | section | 400 | 217 | 68% | 92% | 96% | 0.80 | 0.85 | uc_t23 |
| TF-IDF | fixed | 120 | 631 | 60% | 80% | 84% | 0.69 | 0.74 | uc_t11,uc_t12,uc_t17,uc_t23 |
| TF-IDF | fixed | 220 | 281 | 68% | 76% | 96% | 0.76 | 0.78 | uc_t23 |
| TF-IDF | fixed | 400 | 141 | 72% | 84% | 92% | 0.78 | 0.79 | uc_t17,uc_t23 |
