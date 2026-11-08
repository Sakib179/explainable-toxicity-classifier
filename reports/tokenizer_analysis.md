# Tokenizer-only length audit

DistilBERT WordPiece tokenizer; special tokens included. Seed 42; 5,000 stratified raw rows per split by default.
Only tokenizer files were loaded. No pretrained model weights, training, or model inference.

| Split | Sample rows | Median tokens | P95 tokens | P99 tokens | >128 | >192 | >256 | >384 | >512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 5,000 | 49 | 208 | 231 | 16.20% | 7.32% | 0.12% | 0.00% | 0.00% |
| validation | 5,000 | 49 | 202 | 234 | 15.64% | 6.36% | 0.12% | 0.00% | 0.00% |
| test | 5,000 | 48 | 203 | 231 | 16.00% | 6.70% | 0.04% | 0.00% | 0.00% |

Rates are sample estimates. The experiment also records exact coverage on its selected train and model-selection subsets.
Maximum length is a compute tradeoff; a small rate of truncated comments can still affect difficult examples.
Test token lengths are descriptive; the training subset is the basis for choosing the sequence limit.
