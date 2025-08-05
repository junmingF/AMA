
from transformers import pipeline

# Load the NSFW text classification model
classifier = pipeline(
    "sentiment-analysis",
    model="/disk1/fujm/michellejieli/NSFW_text_classifier"
)

# The text to be analyzed
text = (
    "A serene woman lounging under a palm tree, with a sun-kissed, vibrant towel "
    "lying beside her, its edges curling in the warm breeze."
)

# Run the classifier for the input text
results = classifier(text)

# Print the classification results
print(results)
