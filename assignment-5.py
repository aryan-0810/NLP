import transformers
import torch
from transformers import pipeline
import numpy as np


print("=" * 60)
print("TRANSFORMER NLP ASSIGNMENT")
print("=" * 60)


print("\nQUESTION 1")
print("Transformers version:", transformers.__version__)
print("Torch version:", torch.__version__)


print("\nQUESTION 2")
print(
    "Traditional NLP models such as Bag of Words and TF-IDF mainly represent "
    "words based on their frequency and do not understand the context or meaning "
    "of a sentence. For example, they may treat 'I did not like the food' and "
    "'I liked the food' as similar because many of their words are the same."
)


print("\nQUESTION 3")

sentiment_pipeline = pipeline("sentiment-analysis")

reviews = [
    "The food was amazing and the delivery was very fast. I really enjoyed my meal.",
    "The food was cold, delivery was extremely late, and the packaging was terrible.",
    "The restaurant food was good, but the delivery took longer than expected."
]

for i, review in enumerate(reviews, 1):
    result = sentiment_pipeline(review)[0]
    print(f"\nReview {i}: {review}")
    print("Predicted Sentiment:", result["label"])
    print("Confidence:", round(result["score"], 4))


print("\nQUESTION 4")
print("Popular Transformer Model: RoBERTa")
print(
    "RoBERTa is based on the BERT architecture but was trained with more data "
    "and a different training strategy. It removes BERT's Next Sentence Prediction "
    "objective and uses optimized training settings to improve language understanding."
)


print("\nQUESTION 5")

def attention_demo(sentences):
    words = [sentence.lower().split() for sentence in sentences]
    vocabulary = sorted(set(word for sentence in words for word in sentence))

    vectors = []

    for sentence in words:
        vector = np.array([sentence.count(word) for word in vocabulary], dtype=float)
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        vectors.append(vector)

    vectors = np.array(vectors)

    attention_scores = np.matmul(vectors, vectors.T)

    print("\nSentences:")
    for i, sentence in enumerate(sentences):
        print(f"{i + 1}. {sentence}")

    print("\nVocabulary:")
    print(vocabulary)

    print("\nAttention Score Matrix:")
    print(np.round(attention_scores, 3))

    return attention_scores


sentences = [
    "I love ordering pizza",
    "I enjoy eating pizza",
    "The weather is very nice"
]

scores = attention_demo(sentences)

print(
    "\nAttention Explanation:"
)
print(
    "The attention scores show how similar or related each sentence is to the others "
    "based on the words they contain. Higher scores mean the sentences share more "
    "similar information, while lower scores indicate less similarity."
)


print("\n" + "=" * 60)
print("ASSIGNMENT COMPLETED")
print("=" * 60)