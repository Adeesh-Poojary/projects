import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from spacy_train_data import TRAIN_DATA
import random

# Step 1: Load a blank English model
nlp = spacy.blank("en")

# Step 2: Add NER pipeline
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# Step 3: Add labels from your training data
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Step 4: Train the NER model
def remove_overlaps(data):
    cleaned = []
    for text, ann in data:
        entities = sorted(ann["entities"], key=lambda x: (x[0], x[1]))
        non_overlapping = []
        prev_end = -1
        for start, end, label in entities:
            if start >= prev_end:
                non_overlapping.append((start, end, label))
                prev_end = end
        cleaned.append((text, {"entities": non_overlapping}))
    return cleaned

TRAIN_DATA = remove_overlaps(TRAIN_DATA)
nlp.begin_training()
for i in range(20):  # number of epochs
    random.shuffle(TRAIN_DATA)
    losses = {}
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.5))
    for batch in batches:
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], losses=losses)
    print(f"Epoch {i+1}, Losses: {losses}")

# Step 5: Save the model
nlp.to_disk("resume_ner_model")
print("✅ Model saved to 'resume_ner_model/'")
