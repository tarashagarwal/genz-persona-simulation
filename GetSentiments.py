from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load model & tokenizer
model_name = "minh21/XLNet-Reddit-Sentiment-Analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create pipeline
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Test Reddit-style inputs
examples = [
    "This update is absolute trash, I'm so mad.",
    "Honestly, I kind of love the new design lol.",
    "Meh... I don’t care either way."
]

# Run classification
for text in examples:
    result = classifier(text)[0]
    print(f"Text: {text}\nPrediction: {result['label']}, Confidence: {result['score']:.4f}\n")
