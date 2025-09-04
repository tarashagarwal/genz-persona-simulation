from transformers import pipeline

classifier = pipeline(task="text-classification", 
                      model="SamLowe/roberta-base-go_emotions", 
                      top_k=None, 
                      device=0)

sentences = ["I am the best"]

model_outputs = classifier(sentences)
print(model_outputs[0])
# produces a list of dicts for each of the labels