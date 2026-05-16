from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load trained model
tokenizer = BertTokenizer.from_pretrained("medical_nlp_model")
model = BertForSequenceClassification.from_pretrained("medical_nlp_model")

labels = ["condition", "symptom", "allergy", "medication"]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits).item()
    
    return labels[predicted_class_id]

# Test input
text = "I have arthritis and knee pain and I am allergic to peanuts"

print("Prediction:", predict(text))