import streamlit as st
import subprocess
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score

# Install necessary libraries
with st.spinner("Installing dependencies..."):
    subprocess.run(["pip", "install", "-U", "transformers", "accelerate", "datasets", "bertviz", "umap-learn", "seaborn"])

# Streamlit app title
st.title("IMDB Sentiment Analysis with BERT")

# Load dataset
@st.cache_data()
def load_data():
    return load_dataset("imdb")

imdb_data = load_data()

# Select sample size
sample_size = st.sidebar.slider("Select Sample Size", min_value=100, max_value=1000, value=250, step=50)
train_subset = imdb_data["train"].shuffle(seed=42).select(range(sample_size))
test_subset = imdb_data["test"].shuffle(seed=42).select(range(sample_size))

# Initialize tokenizer
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

train_subset = train_subset.map(tokenize, batched=True)
test_subset = test_subset.map(tokenize, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

# Define training arguments
training_args = TrainingArguments(
    output_dir="bert_imdb_sentiment",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    evaluation_strategy="no",
    save_strategy="no",
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=False,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_subset,
    eval_dataset=test_subset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

if st.button("Train Model"):
    with st.spinner("Training in progress..."):
        trainer.train()
    st.success("Training Completed!")
