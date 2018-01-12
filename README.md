# qa-system
Final Project for 6.864 Advanced Natural Language Processing

Part 1: Question Retrieval
- train_lstm.py for training encoder
- eval_lstm.py for evaluating trained models
- lstm.py defines model used in train_lstm
- maxmarginloss.py oddly named for cosine similarity calculation
- load_data.py to create batches for Ubuntu dataset, prune_glove.py for creating glove dictionary, and word2vec.py for train_lstm
- metrics.py to calculate MAP, MRR, P@1, P@5 in eval_lstm

Part 2: Transfer Learning
- tfidf_baseline.py for tfidf baseline score calculation
- domain_adaptation.py for domain adaptation training
- domain_classifier.py defines feed forward network for domain classification in domain_adaptation
- eval_lstm.py to evaluate direct_transfer and domain_adaptation models
- metrics.py and meter.py for metrics calculations in eval_lstm
- load_android_data.py and load_android_eval.py to prepare data for Android dataset
- dan.py is an unfinished attempt at other adversarial techniques