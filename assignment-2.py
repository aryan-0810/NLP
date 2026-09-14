import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

def lowercase_text(text):
    return text.lower()


def remove_punctuation(text):
    cleaned_text = ""

    for character in text:
        if character not in string.punctuation:
            cleaned_text += character

    return cleaned_text


def tokenize_text(text):
    return text.split()


def remove_stopwords(message):
    stop_words = set(stopwords.words("english"))
    words = message.split()

    filtered_words = []

    for word in words:
        if word.lower() not in stop_words:
            filtered_words.append(word)

    return " ".join(filtered_words)


def stem_hashtags(hashtags):
    stemmer = PorterStemmer()

    print(f"{'Original':<20}{'Stemmed':<20}")
    print("-" * 40)

    for hashtag in hashtags:
        word = hashtag.replace("#", "")
        stemmed_word = stemmer.stem(word)

        print(f"{hashtag:<20}{'#' + stemmed_word:<20}")


def question_1():
    print("\n" + "=" * 70)
    print("1. INSTAGRAM BIO LOWERCASE")
    print("=" * 70)

    bio = input("Enter your Instagram bio: ")
    result = lowercase_text(bio)

    print("Lowercase Bio:")
    print(result)


def question_2():
    print("\n" + "=" * 70)
    print("2. FLIPKART REVIEW PUNCTUATION REMOVAL")
    print("=" * 70)

    review = input("Enter a Flipkart product review: ")
    result = remove_punctuation(review)

    print("Cleaned Review:")
    print(result)


def question_3():
    print("\n" + "=" * 70)
    print("3. ZOMATO FOOD DESCRIPTION TOKENIZATION")
    print("=" * 70)

    description = "Delicious spicy paneer pizza with fresh vegetables and cheese"

    tokens = tokenize_text(description)

    print("Food Description:")
    print(description)

    print("\nTokens:")
    print(tokens)


def question_4():
    print("\n" + "=" * 70)
    print("4. WHATSAPP STOPWORD REMOVAL")
    print("=" * 70)

    message = input("Enter a WhatsApp message: ")
    result = remove_stopwords(message)

    print("Message after removing stopwords:")
    print(result)


def question_5():
    print("\n" + "=" * 70)
    print("5. TWITTER HASHTAG STEMMING")
    print("=" * 70)

    hashtags = ["#playing", "#dancing", "#eating", "#running", "#studying"]

    print("Original and Stemmed Hashtags:")
    stem_hashtags(hashtags)


def main():
    print("=" * 70)
    print("NLP PREPROCESSING ASSIGNMENT")
    print("=" * 70)
    print("Name: Aryan Pandya")
    print("Language: Python")

    question_1()
    question_2()
    question_3()
    question_4()
    question_5()

    print("\n" + "=" * 70)
    print("ASSIGNMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()