| Embedder | Chunker | Size (words) | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR | AUROC | Missed |
|---|---|---|---|---|---|---|---|---|---|
| tfidf | fixed | 120 | 251 | 70% | 78% | 83% | 0.74 | 0.92 | t15,t16,t18,t20 |
| tfidf | fixed | 220 | 118 | 70% | 91% | 91% | 0.79 | 0.91 | t18,t20 |
| tfidf | fixed | 400 | 64 | 83% | 91% | 96% | 0.87 | 0.92 | t20 |
| tfidf | section | 120 | 254 | 65% | 83% | 91% | 0.76 | 0.91 | t16,t20 |
| tfidf | section | 220 | 178 | 65% | 91% | 91% | 0.78 | 0.88 | t16,t20 |
| tfidf | section | 400 | 148 | 70% | 91% | 96% | 0.81 | 0.86 | t20 |
