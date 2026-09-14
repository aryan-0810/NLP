import string
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def question_1():
    print("\n" + "=" * 80)
    print("1. INSTAGRAM CAPTIONS - VOCABULARY")
    print("=" * 80)

    captions = [
        "Amazing day with amazing friends!",
        "Exploring new places and creating memories.",
        "Good vibes, great friends, beautiful memories!"
    ]

    vocabulary = set()

    for caption in captions:
        cleaned_caption = clean_text(caption)
        words = cleaned_caption.split()
        vocabulary.update(words)

    vocabulary = sorted(vocabulary)

    print("Instagram Captions:")
    for caption in captions:
        print(caption)

    print("\nUnique Vocabulary:")
    print(vocabulary)

    print("\nTotal Unique Words:", len(vocabulary))


def question_2():
    print("\n" + "=" * 80)
    print("2. ZOMATO REVIEWS - BAG OF WORDS")
    print("=" * 80)

    reviews = [
        "Loved the pizza",
        "Service was slow",
        "Great ambience"
    ]

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(reviews)

    print("Reviews:")
    for review in reviews:
        print(review)

    print("\nVocabulary:")
    print(vectorizer.get_feature_names_out())

    print("\nBag of Words Matrix:")
    print(matrix.toarray())


def question_3():
    print("\n" + "=" * 80)
    print("3. WHATSAPP WORD FREQUENCY")
    print("=" * 80)

    messages = [
        "The movie was really good",
        "The food is really good",
        "The service is slow",
        "Good food and good service",
        "The movie is good"
    ]

    stopwords = {"the", "is", "at"}

    word_counts = Counter()

    for message in messages:
        words = message.lower().split()

        for word in words:
            word = word.strip(string.punctuation)

            if word and word not in stopwords:
                word_counts[word] += 1

    print("WhatsApp Messages:")
    for message in messages:
        print(message)

    print("\nWord Frequency:")

    for word, count in word_counts.most_common():
        print(f"{word}: {count}")


def question_4():
    print("\n" + "=" * 80)
    print("4. FLIPKART REVIEWS - TF-IDF")
    print("=" * 80)

    reviews = [
        "The product has excellent quality",
        "The quality of the product is good",
        "Poor quality but the design is attractive"
    ]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(reviews)

    feature_names = vectorizer.get_feature_names_out()

    quality_index = list(feature_names).index("quality")

    print("Flipkart Reviews:")
    for i, review in enumerate(reviews, start=1):
        print(f"Review {i}: {review}")

    print("\nTF-IDF value for the word 'quality':")

    for i, row in enumerate(matrix.toarray(), start=1):
        print(f"Review {i}: {row[quality_index]:.4f}")


def question_5():
    print("\n" + "=" * 80)
    print("5. BAG OF WORDS VS TF-IDF")
    print("=" * 80)

    print("""
Difference 1:
Bag of Words mainly counts how many times a word appears in a document.
TF-IDF considers both the frequency of a word in a document and how common
that word is across the entire collection of documents.

Difference 2:
Bag of Words gives similar importance to words based mainly on their count,
while TF-IDF gives lower importance to words that appear in many documents
and higher importance to words that are more specific to a particular document.

Real-World Example:
Consider a search feature in a shopping application such as Flipkart.
Suppose a user searches for "high quality wireless headphones".

Words such as "the", "product", or "quality" may occur in many product
descriptions. TF-IDF can help give more importance to terms that are
specific to particular products and less importance to very common words.

Therefore, TF-IDF can be more useful than simple Bag of Words when we want
to identify important and distinctive words in documents, such as product
search, document search, or review analysis.
""")


def main():
    print("=" * 80)
    print("NLP VECTORIZATION ASSIGNMENT")
    print("=" * 80)
    print("Name: Aryan Pandya")
    print("Language: Python")

    question_1()
    question_2()
    question_3()
    question_4()
    question_5()

    print("\n" + "=" * 80)
    print("ASSIGNMENT COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()