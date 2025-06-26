# Sentiment-Analysis
Sentiment Analysis by fine tuning a pretrained model on IMDB review dataset

# Sentiment Analysis with DistilBERT

## Overview

This project fine-tunes DistilBERT on a sentiment analysis IMDB review dataset to classify text reviews as positive or negative. The model is trained using the Hugging Face Transformers library and deployed as a web application.

## Features

- Fine-tuned DistilBERT using full parameter training

- Achieved 91.3% accuracy on the test dataset

- Evaluation Metrics: Precision, Recall, F1-Score, Loss

- Deployment: Hosted as a web app for real-time sentiment analysis

## Dataset

The dataset consists of text reviews labeled as positive (1) or negative (0).

- Train-Test Split: 70-30 (Stratified)

- Training Samples: 7000

- Validation Samples: 1500

- Test Samples: 1500

## Performance Metrics

| Metric           | Value  |
|-----------------|--------|
| **Accuracy**     | 91.3%  |
| **Precision**    | 91%    |
| **Recall**       | 91%    |
| **F1-Score**     | 0.91   |
| **Test Loss**    | 0.273  |
| **Test Runtime** | 161.6s |
| **Samples/Sec**  | 61.9   |


## Future Improvements

- Implement LoRA for memory-efficient fine-tuning

- Expand dataset for multi-class sentiment analysis

- Optimize model inference speed

