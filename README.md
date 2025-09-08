# NLP_CourseFinalProject
Repository for the final project of "Practical Machine Learning for Natural Language Processing" (University of Vienna, Summer Semester 2024).

# Project Overview
This project presents a comparative analysis of modern Natural Language Processing (NLP) methodologies for the task of detecting sexist language in tweets. It evaluates the performance and cost of four distinct approaches:

  1. Fine-Tuned BERT Model for sequence classification. ('germeval.ipynb')
  2. Zero-Shot and Few-Shot Prompting using DeepSeek and GPT APIs. ('comparisonAPI.ipynb')
  3. Retrieval-Augmented Generation (RAG) using a VectorStoreIndex. ('comparisonVectorStoreIndex.ipynb')
  4. Retrieval-Augmented Generation (RAG) using a KeywordTableIndex. ('comparisonKeywordTableIndex.ipynb')

# Motivation 
Detecting sexist language is a key challenge in studying online discourse. Automated methods can help researchers analyze large-scale corpora of social media data to understand the prevalence and nature of sexist rhetoric. However, the nuanced and context-dependent nature of such language makes it a difficult classification problem. 

# Methodology
The performance of each method was evaluated and compared on a standardized dataset (with dedicated train, development, and test set) using the F1-score as the primary metric. The implementation details for each approach are as follows:

  1. Fine-Tuned BERT (bert-base-uncased): Implemented using the Hugging Face transformers library, optimized for sequence classification on our specific dataset.
  2. API Prompting (DeepSeek & GPT): Leveraged zero-shot and few-shot prompting strategies to query large language models via their APIs for classification, testing their ability to generalize with little to no task-specific training.
  3. RAG Pipeline (VectorStoreIndex): Implemented a RAG system using LlamaIndex's VectorStoreIndex to retrieve the most semantically similar examples from the training set to provide context for an LLM's classification decision.
  4. RAG Pipeline (KeywordTableIndex): Implemented an alternative RAG system using a KeywordTableIndex to retrieve examples based on keyword matching, providing a different retrieval strategy for comparison.
  5. RAG Pipelines (3. and 4.) combined with API Prompting (DeepSeek & GPT) (2.).

# Findings
Best overall performance was achieved by models relying on the API (specifically DeepSeek) both with and without RAG. The fine-tuned BERT models achieved average results, while models relying only on RAG yielded the lowest F1-scores.


