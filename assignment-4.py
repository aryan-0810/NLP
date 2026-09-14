import os
import urllib.request
import zipfile
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def question_1():
    print("\n" + "=" * 80)
    print("1. IPL TWEET DATASET")
    print("=" * 80)

    dataset_url = "https://www.kaggle.com/api/v1/datasets/download/sripaadsrinivasan/ipl-2020-tweets"
    zip_file = "ipl_tweets.zip"
    extract_folder = "ipl_tweets_data"

    if not os.path.exists(extract_folder):
        print("Downloading IPL Twitter dataset...")
        urllib.request.urlretrieve(dataset_url, zip_file)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

    csv_file = None

    for root, directories, files in os.walk(extract_folder):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_file = os.path.join(root, file)
                break
        if csv_file:
            break

    if csv_file is None:
        print("CSV file was not found.")
        return

    tweets_df = pd.read_csv(csv_file)

    print("Dataset file:")
    print(csv_file)

    print("\nDataset shape:")
    print(tweets_df.shape)

    print("\nFirst 10 tweets:")
    print(tweets_df.head(10).to_string())


def create_zomato_dataset():
    reviews = [
        "The pizza was amazing and delicious",
        "Loved the food and excellent service",
        "Great taste and wonderful ambience",
        "The food was fresh and tasty",
        "Amazing restaurant with great food",
        "Excellent service and delicious pizza",
        "I loved the food",
        "The restaurant was fantastic",
        "Very tasty food and good service",
        "Great experience and delicious food",
        "The food was terrible",
        "Very bad service and cold food",
        "The pizza was horrible",
        "Worst restaurant experience",
        "The food was tasteless and bad",
        "Very poor service",
        "I hated the food",
        "The restaurant was disappointing",
        "Bad taste and terrible service",
        "The food quality was horrible"
    ]

    labels = [
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative"
    ]

    return reviews, labels


def question_2():
    print("\n" + "=" * 80)
    print("2. ZOMATO REVIEW SENTIMENT USING NAIVE BAYES")
    print("=" * 80)

    reviews, labels = create_zomato_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        reviews,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )

    vectorizer = CountVectorizer()

    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_vectorized, y_train)

    predictions = model.predict(x_test_vectorized)

    accuracy = accuracy_score(y_test, predictions)

    print("Number of reviews:", len(reviews))
    print("Training reviews:", len(x_train))
    print("Testing reviews:", len(x_test))

    print("\nActual Labels:")
    print(y_test)

    print("\nPredicted Labels:")
    print(predictions)

    print(f"\nNaive Bayes Accuracy: {accuracy:.2f}")

    test_reviews = [
        "The food was delicious and amazing",
        "The service was horrible and bad"
    ]

    test_vectors = vectorizer.transform(test_reviews)
    test_predictions = model.predict(test_vectors)

    print("\nNew Review Predictions:")

    for review, prediction in zip(test_reviews, test_predictions):
        print(f"{review} -> {prediction}")


def question_3():
    print("\n" + "=" * 80)
    print("3. WHATSAPP SPAM DETECTOR USING LOGISTIC REGRESSION")
    print("=" * 80)

    messages = [
        "Congratulations you won a free prize",
        "You have won a free lottery ticket",
        "Click this link to claim your free reward",
        "Win cash now by entering this contest",
        "You are selected for a free gift",
        "Claim your free vacation today",
        "Congratulations win a brand new phone",
        "Free recharge available click now",
        "Win exciting rewards by clicking here",
        "You won a free shopping voucher",
        "Hey are we meeting today",
        "Please send me the notes",
        "Can you call me when you are free",
        "The meeting is scheduled for tomorrow",
        "I will reach home by eight",
        "Let's have lunch together",
        "Please bring your laptop tomorrow",
        "Can you send me the project file",
        "I am going to the market",
        "See you at the college tomorrow"
    ]

    labels = [
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam",
        "not spam"
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        messages,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )

    vectorizer = TfidfVectorizer()

    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_vectorized, y_train)

    predictions = model.predict(x_test_vectorized)

    accuracy = accuracy_score(y_test, predictions)

    print("Training messages:", len(x_train))
    print("Testing messages:", len(x_test))
    print(f"Logistic Regression Accuracy: {accuracy:.2f}")

    custom_messages = [
        "Congratulations you won a free smartphone",
        "Can you send me the assignment file",
        "You can win a free shopping voucher today"
    ]

    custom_vectors = vectorizer.transform(custom_messages)
    custom_predictions = model.predict(custom_vectors)

    print("\nPredictions for 3 New Messages:")

    for message, prediction in zip(custom_messages, custom_predictions):
        print(f"{message}")
        print(f"Prediction: {prediction}\n")


def question_4():
    print("\n" + "=" * 80)
    print("4. NAIVE BAYES VS LOGISTIC REGRESSION")
    print("=" * 80)

    reviews, labels = create_zomato_dataset()

    x_train, x_test, y_train, y_test = train_test_split(
        reviews,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels
    )

    vectorizer = TfidfVectorizer()

    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(x_train_vectorized, y_train)

    naive_bayes_predictions = naive_bayes_model.predict(x_test_vectorized)

    naive_bayes_accuracy = accuracy_score(
        y_test,
        naive_bayes_predictions
    )

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(x_train_vectorized, y_train)

    logistic_predictions = logistic_model.predict(x_test_vectorized)

    logistic_accuracy = accuracy_score(
        y_test,
        logistic_predictions
    )

    print(f"Naive Bayes Accuracy: {naive_bayes_accuracy:.2f}")
    print(f"Logistic Regression Accuracy: {logistic_accuracy:.2f}")

    print("\nModel Comparison:")

    if naive_bayes_accuracy > logistic_accuracy:
        print("Naive Bayes performs better on this test split.")
    elif logistic_accuracy > naive_bayes_accuracy:
        print("Logistic Regression performs better on this test split.")
    else:
        print("Both models have the same accuracy on this test split.")


def question_5():
    print("\n" + "=" * 80)
    print("5. FLIPKART PRODUCT REVIEW CLASSIFICATION")
    print("=" * 80)

    prompt = """
Generate Python code using scikit-learn to classify Flipkart product reviews
as positive or negative using Logistic Regression. Use TF-IDF for text
vectorization, train the model on sample reviews, and test it on two new
reviews.
"""

    print("Prompt used with ChatGPT:")
    print(prompt)

    reviews = [
        "Excellent product and very good quality",
        "Amazing quality and fast delivery",
        "The product is excellent and worth the money",
        "Very happy with this purchase",
        "Great product and good performance",
        "I really liked this product",
        "The quality is amazing",
        "Very satisfied with the product",
        "Product works perfectly",
        "Good product and excellent quality",
        "Very poor quality product",
        "The product stopped working",
        "Terrible product and bad quality",
        "Very disappointed with this purchase",
        "Worst product I have bought",
        "Poor quality and slow performance",
        "The product is useless",
        "Bad experience with this product",
        "Very poor product quality",
        "I am unhappy with this purchase"
    ]

    labels = [
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "positive",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative",
        "negative"
    ]

    vectorizer = TfidfVectorizer()

    x_train, x_test, y_train, y_test = train_test_split(
        reviews,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )

    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(x_train_vectorized, y_train)

    predictions = model.predict(x_test_vectorized)

    accuracy = accuracy_score(y_test, predictions)

    print("Generated classification code was implemented using:")
    print("TF-IDF Vectorization + Logistic Regression")

    print(f"\nModel Accuracy: {accuracy:.2f}")

    sample_reviews = [
        "The product quality is excellent and I am very happy",
        "The product quality is very poor and disappointing"
    ]

    sample_vectors = vectorizer.transform(sample_reviews)
    sample_predictions = model.predict(sample_vectors)

    print("\nTest Results:")

    for review, prediction in zip(sample_reviews, sample_predictions):
        print(f"Review: {review}")
        print(f"Prediction: {prediction}")
        print()


def main():
    print("=" * 80)
    print("NLP TEXT CLASSIFICATION ASSIGNMENT")
    print("=" * 80)
    print("Name: Aryan Pandya")
    print("Language: Python")

    question_1()
    question_2()
    question_3()
    question_4()
    question_5()

    print("=" * 80)
    print("ASSIGNMENT COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()