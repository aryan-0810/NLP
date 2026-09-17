import re
import nltk
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)


def section_a():
    print("\n" + "=" * 70)
    print("SECTION A - CONCEPT APPLICATION")
    print("=" * 70)

    print("""
QUESTION 1

Two important NLP techniques for the customer support chatbot are text
preprocessing and text classification.

Text preprocessing is necessary to clean and prepare customer messages
before the chatbot processes them. It can include converting text to
lowercase, removing unnecessary punctuation, tokenisation, stopword
removal and stemming or lemmatization. Without preprocessing, the same
meaning may appear in many different forms and the chatbot may have
difficulty identifying the important words.

Text classification is necessary to identify the purpose or category of
the customer's message. For example, "Where is my order?" can be
classified as a delivery tracking query, while "I want to cancel my
order" can be classified as an order cancellation query.

If preprocessing is removed, irrelevant words and different word forms
can reduce the quality of the input. If classification is removed, the
chatbot may understand individual words but may not correctly route the
customer to the required service.
""")

    print("""
QUESTION 2

The preprocessing pipeline for the 50,000 customer reviews would start
by cleaning the raw text. First, I would convert all text to lowercase
so that words such as "Food" and "food" are treated consistently.

Next, I would remove unnecessary punctuation and handle special
characters and numbers according to the requirements of the NLP task.
The cleaned text would then be tokenised into individual words.

After tokenisation, English stopwords such as "the", "is", and "a" can
be removed because they usually provide limited information for many
text classification tasks. Finally, stemming or lemmatization can be
applied to reduce related words to a common form.

The order is important because tokenisation should be performed after
basic text cleaning, and stopword removal and stemming are applied to
the resulting tokens.

If stopword removal is skipped, common words can create unnecessary
features in the dataset. This can increase the size of the feature
space and may make it harder for the model to focus on words that are
more useful for predicting the required outcome.
""")

    print("""
QUESTION 3

Bag of Words represents text by counting how frequently words occur in
each document. It is simple and useful, but common words can receive
high importance simply because they occur frequently.

TF-IDF also represents text numerically but gives higher importance to
words that are important in a particular document and less importance
to words that occur across many documents.

For the menu search ranking task, I would choose TF-IDF because it can
give more importance to terms that distinguish one menu description
from another. For example, words such as "spicy" or "chicken" can help
match a customer's query with relevant menu descriptions.

One limitation of TF-IDF is that it does not understand the deeper
meaning or context of words. It mainly represents words using numerical
importance scores, so phrases with similar meanings may not always be
recognised as semantically similar.
""")

    print("""
QUESTION 4

Naive Bayes assumes that the input features are conditionally
independent given the class.

For example, when classifying a support ticket, the model may consider
words such as "cold", "late", "food", and "delivery" as individual
features. The Naive Bayes assumption treats these features as
independent when calculating the probability of a class.

In a realistic food delivery example, the words "cold" and "food" may
be strongly related when a customer says that the food arrived cold.
Similarly, "late delivery" is a meaningful combination of words.

Therefore, the independence assumption can be violated because words
can depend on each other and their meaning can change based on the
surrounding words. Although Naive Bayes can still work well despite
this assumption, strong dependencies between features can reduce
classification accuracy in some situations.
""")

    print("""
QUESTION 5

Since the dataset contains 80% Positive, 15% Neutral, and 5% Negative
reviews, accuracy alone may be misleading. A model could obtain high
accuracy by mostly predicting the majority Positive class while
performing poorly on the smaller classes.

Two metrics I would prioritise are precision and recall. Precision
measures how many predictions for a class are actually correct, while
recall measures how many actual examples of a class the model is able
to identify.

F1-score can also be useful because it combines precision and recall.

One technique I would apply before training is oversampling the
minority classes. This can increase the representation of Neutral and
Negative reviews in the training data and help the model learn those
classes better.
""")

    print("""
QUESTION 6

The attention mechanism in Transformers allows the model to consider
relationships between different words in a sequence, even when those
words are far apart.

Traditional RNN-based models process sequences step by step. With long
sequences, information from earlier words can become difficult to retain.
Transformers use attention to directly relate different parts of the
sequence and can process many positions more efficiently.

For example, in a long customer review, the model can use attention to
connect a word near the beginning of the review with an important word
much later in the sentence.

One practical challenge of using Transformers at several hundred
requests per minute is high computational and memory usage. This can
increase response time and infrastructure cost.

One common way to mitigate this challenge is to use a smaller or
optimized Transformer model and deploy it using suitable hardware or
model optimization techniques such as batching.
""")


def section_b_task1():
    print("\n" + "=" * 70)
    print("SECTION B - TASK 1: FOOD REVIEW TEXT PREPROCESSOR")
    print("=" * 70)

    reviews = [
        "The food was amazing and arrived very hot!",
        "My order was late and the pizza was cold.",
        "The delivery app is easy to use and very helpful."
    ]

    for review in reviews:
        print("\nOriginal Review:", review)
        print("Processed Review:", preprocess_text(review))


def section_b_task2():
    print("\n" + "=" * 70)
    print("SECTION B - TASK 2: TF-IDF VECTORISER FOR MENU REVIEWS")
    print("=" * 70)

    reviews = [
        "The pizza was hot and delicious",
        "The pizza arrived cold and late",
        "Amazing burger with fresh ingredients",
        "The burger was cold and tasteless",
        "Fast delivery and excellent service",
        "Delivery was very slow and disappointing",
        "The food was okay and average",
        "Good food with fast delivery"
    ]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(reviews)
    feature_names = vectorizer.get_feature_names_out()

    df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=[f"Review {i + 1}" for i in range(len(reviews))]
    )

    print("\nFULL TF-IDF MATRIX:")
    print(df.round(3))

    print("\nTOP 3 TERMS FOR EACH REVIEW:")

    for i in range(len(reviews)):
        scores = tfidf_matrix[i].toarray().flatten()
        top_indices = scores.argsort()[-3:][::-1]

        print("\nReview", i + 1)
        for index in top_indices:
            print(feature_names[index], ":", round(scores[index], 3))


def section_b_task3():
    print("\n" + "=" * 70)
    print("SECTION B - TASK 3: COMPLAINT CATEGORY CLASSIFIER")
    print("=" * 70)

    complaints = [
        "My order arrived very late",
        "The delivery rider was late",
        "My food has not been delivered",
        "The delivery is taking too long",
        "My order is still waiting for delivery",
        "The rider could not find my address",
        "The delivery person arrived late",
        "My order was delayed",
        "The food arrived cold",
        "My pizza was burnt",
        "The burger tasted terrible",
        "The food quality was very poor",
        "My meal was cold and tasteless",
        "The chicken was undercooked",
        "The food was stale",
        "The pizza was overcooked",
        "The app is not opening",
        "I cannot login to the app",
        "The payment failed in the app",
        "The app keeps crashing",
        "I cannot place an order",
        "The application is showing an error",
        "My payment is not working",
        "The app is very slow"
    ]

    labels = [
        "Delivery", "Delivery", "Delivery", "Delivery",
        "Delivery", "Delivery", "Delivery", "Delivery",
        "Food Quality", "Food Quality", "Food Quality", "Food Quality",
        "Food Quality", "Food Quality", "Food Quality", "Food Quality",
        "App", "App", "App", "App",
        "App", "App", "App", "App"
    ]

    processed = [preprocess_text(text) for text in complaints]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("CONFUSION MATRIX:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["Delivery", "Food Quality", "App"]
    ))


def section_b_task4():
    print("\n" + "=" * 70)
    print("SECTION B - TASK 4: SENTIMENT ANALYSIS MODEL COMPARISON")
    print("=" * 70)

    reviews = [
        "The food was excellent and delicious",
        "Amazing food and fast delivery",
        "I loved the pizza",
        "The burger was fresh and tasty",
        "Great service and good food",
        "The delivery was quick and perfect",
        "Very happy with my order",
        "The food tasted wonderful",
        "Excellent restaurant and service",
        "The meal was fresh and hot",
        "Really good food",
        "The order arrived quickly",
        "The food was amazing",
        "I enjoyed the meal",
        "Very tasty and satisfying",
        "The food was terrible",
        "The delivery was very late",
        "My pizza was cold",
        "The burger tasted bad",
        "Very poor food quality",
        "The order arrived late",
        "I did not like the food",
        "The meal was cold and tasteless",
        "The service was terrible",
        "My food was burnt",
        "Very disappointing experience",
        "The delivery was extremely slow",
        "The food was stale",
        "I am unhappy with the order",
        "The meal was awful"
    ]

    labels = [
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative"
    ]

    processed_reviews = [preprocess_text(review) for review in reviews]

    X_train, X_test, y_train, y_test = train_test_split(
        processed_reviews,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", MultinomialNB())
    ])

    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    nb_pipeline.fit(X_train, y_train)
    lr_pipeline.fit(X_train, y_train)

    nb_predictions = nb_pipeline.predict(X_test)
    lr_predictions = lr_pipeline.predict(X_test)

    results = [
        [
            "Naive Bayes",
            accuracy_score(y_test, nb_predictions),
            precision_score(y_test, nb_predictions, pos_label="Positive"),
            recall_score(y_test, nb_predictions, pos_label="Positive"),
            f1_score(y_test, nb_predictions, pos_label="Positive")
        ],
        [
            "Logistic Regression",
            accuracy_score(y_test, lr_predictions),
            precision_score(y_test, lr_predictions, pos_label="Positive"),
            recall_score(y_test, lr_predictions, pos_label="Positive"),
            f1_score(y_test, lr_predictions, pos_label="Positive")
        ]
    ]

    comparison = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    )

    print("\nMODEL COMPARISON:")
    print(comparison.round(3))

    # I would deploy Logistic Regression if it gives better overall test metrics because it provides strong text classification performance.


def mini_project():
    print("\n" + "=" * 70)
    print("SECTION C - FOOD DELIVERY REVIEW INTELLIGENCE SYSTEM")
    print("=" * 70)

    sentiment_texts = [
        "The food was excellent and delivery was fast",
        "Amazing pizza and great service",
        "The meal was fresh and delicious",
        "Very tasty food and quick delivery",
        "I loved the food",
        "Excellent experience with my order",
        "The food was wonderful",
        "Very good service and fresh food",
        "The order arrived perfectly",
        "Great food and fast delivery",
        "The meal was amazing",
        "I enjoyed my order",
        "The food was terrible and cold",
        "My order arrived very late",
        "The pizza was burnt",
        "The delivery was extremely slow",
        "The food tasted bad",
        "Very poor experience",
        "My meal was cold",
        "The order was disappointing",
        "The food was stale",
        "The rider was very late",
        "The app stopped working",
        "I am unhappy with the service"
    ]

    sentiment_training_labels = [
        "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative"
    ]

    issue_texts = [
        "My order arrived late",
        "The delivery rider was delayed",
        "My food has not arrived",
        "The delivery was very slow",
        "The rider could not find my address",
        "My order is still waiting",
        "The food arrived cold",
        "My pizza was burnt",
        "The burger was tasteless",
        "The food quality was poor",
        "My chicken was undercooked",
        "The meal was stale",
        "The app is not working",
        "I cannot login to the app",
        "The payment failed",
        "The application keeps crashing",
        "I cannot place an order",
        "The app shows an error",
        "I have a question about the restaurant",
        "I want information about an offer",
        "I have a general question",
        "I need help with my account",
        "I want to know more about the service",
        "I have another issue"
    ]

    issue_training_labels = [
        "Delivery", "Delivery", "Delivery", "Delivery",
        "Delivery", "Delivery",
        "Food Quality", "Food Quality", "Food Quality",
        "Food Quality", "Food Quality", "Food Quality",
        "App", "App", "App", "App", "App", "App",
        "General", "General", "General", "General",
        "General", "General"
    ]

    stored_reviews = []

    sentiment_processed = [preprocess_text(text) for text in sentiment_texts]
    sentiment_vectorizer = TfidfVectorizer()
    sentiment_features = sentiment_vectorizer.fit_transform(sentiment_processed)
    sentiment_model = MultinomialNB()
    sentiment_model.fit(sentiment_features, sentiment_training_labels)

    issue_processed = [preprocess_text(text) for text in issue_texts]
    issue_vectorizer = TfidfVectorizer()
    issue_features = issue_vectorizer.fit_transform(issue_processed)
    issue_model = MultinomialNB()
    issue_model.fit(issue_features, issue_training_labels)

    def classify_review(text):
        if len(stored_reviews) < 1:
            print("\nNo reviews have been added yet.")
            return

        processed = preprocess_text(text)
        sentiment_features = sentiment_vectorizer.transform([processed])
        issue_features = issue_vectorizer.transform([processed])

        sentiment = sentiment_model.predict(sentiment_features)[0]
        issue = issue_model.predict(issue_features)[0]

        print("\nClassification Result")
        print("-" * 40)
        print("Original Review:", text)
        print("Processed Review:", processed)
        print("Predicted Sentiment:", sentiment)
        print("Predicted Issue Category:", issue)

    def add_review():
        text = input("\nEnter customer review: ")

        if text.strip() == "":
            print("Review cannot be empty.")
            return

        processed = preprocess_text(text)

        stored_reviews.append({
            "original": text,
            "processed": processed
        })

        print("\nReview added successfully.")
        print("Processed Review:", processed)

    def summary_report():
        print("\n" + "=" * 50)
        print("SUMMARY REPORT")
        print("=" * 50)

        print("Total Reviews Added:", len(stored_reviews))

        if len(stored_reviews) == 0:
            print("No reviews available.")
            return

        sentiments = []
        issues = []
        words = []

        for review in stored_reviews:
            processed = review["processed"]

            sentiment_features = sentiment_vectorizer.transform([processed])
            issue_features = issue_vectorizer.transform([processed])

            sentiment = sentiment_model.predict(sentiment_features)[0]
            issue = issue_model.predict(issue_features)[0]

            sentiments.append(sentiment)
            issues.append(issue)
            words.extend(processed.split())

        sentiment_count = Counter(sentiments)
        issue_count = Counter(issues)
        word_count = Counter(words)

        print("\nSentiment Count:")
        print("Positive:", sentiment_count["Positive"])
        print("Negative:", sentiment_count["Negative"])

        print("\nIssue Category Count:")
        print("Delivery:", issue_count["Delivery"])
        print("Food Quality:", issue_count["Food Quality"])
        print("App:", issue_count["App"])
        print("General:", issue_count["General"])

        print("\nTop 5 Content Words:")

        for word, count in word_count.most_common(5):
            print(word, ":", count)

    while True:
        print("\n" + "=" * 50)
        print("FOOD DELIVERY REVIEW INTELLIGENCE SYSTEM")
        print("=" * 50)
        print("1. Add a new review")
        print("2. Classify a review")
        print("3. View a summary report")
        print("4. Exit")
        print("=" * 50)

        choice = input("Enter your choice: ")

        if choice == "1":
            add_review()

        elif choice == "2":
            if len(stored_reviews) < 1:
                print("\nMinimum training samples are not available for classification.")
                print("Please add at least one review first.")
            else:
                text = input("\nEnter review to classify: ")

                if text.strip() == "":
                    print("Review cannot be empty.")
                else:
                    stored_reviews.append({
                        "original": text,
                        "processed": preprocess_text(text)
                    })
                    classify_review(text)

        elif choice == "3":
            summary_report()

        elif choice == "4":
            print("\nExiting Mini Project.")
            break

        else:
            print("\nInvalid choice. Please select 1, 2, 3, or 4.")


def section_d():
    print("\n" + "=" * 70)
    print("SECTION D - AI-AUGMENTED LEARNING")
    print("=" * 70)

    print("""
STEP 1 - AI PROMPT

Write a simple Python 3 program for an NLP assessment based on a food
delivery company. The program should accept a list of food delivery
customer reviews, preprocess each review using tokenisation, English
stopword removal and PorterStemmer, create a TF-IDF matrix, and train a
Logistic Regression model to classify reviews as Positive or Negative.
Split the data into training and testing sets, print the model accuracy
on the test data, and then allow the user to enter three new food
delivery reviews at runtime and predict whether each review is Positive
or Negative. Keep the code beginner-friendly and suitable for running
in VS Code.
""")

    reviews = [
        "The food was amazing",
        "The delivery was excellent",
        "I loved the pizza",
        "The burger was delicious",
        "Great food and fast service",
        "Very tasty meal",
        "The order was perfect",
        "Excellent restaurant",
        "The food was fresh",
        "Very good experience",
        "The food was terrible",
        "The delivery was very late",
        "The pizza was cold",
        "The burger tasted bad",
        "Very poor service",
        "The food was disappointing",
        "The order was awful",
        "The meal was tasteless",
        "The delivery was slow",
        "I hated the food"
    ]

    labels = [
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Positive", "Positive", "Positive", "Positive", "Positive",
        "Negative", "Negative", "Negative", "Negative", "Negative",
        "Negative", "Negative", "Negative", "Negative", "Negative"
    ]

    processed_reviews = [preprocess_text(review) for review in reviews]

    X_train, X_test, y_train, y_test = train_test_split(
        processed_reviews,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\nAI-ASSISTED LOGISTIC REGRESSION MODEL")
    print("Model Accuracy:", round(accuracy, 3))

    print("""
STEP 2 - TEST AND DEBUG NOTE

The original AI-generated version did not remove punctuation during
preprocessing. I added punctuation removal and extra whitespace handling
so that similar words are processed consistently. I also used a
stratified train-test split so that both Positive and Negative classes
are represented in the test data. These changes make preprocessing and
model evaluation more reliable.
""")

    print("Enter three new reviews:")

    for i in range(3):
        new_review = input("\nEnter review " + str(i + 1) + ": ")

        if new_review.strip() == "":
            print("Please enter a valid review.")
            continue

        processed = preprocess_text(new_review)
        features = vectorizer.transform([processed])
        prediction = model.predict(features)[0]

        print("Predicted Sentiment:", prediction)


def main():
    while True:
        print("\n")
        print("=" * 70)
        print("M11-A1 - NATURAL LANGUAGE PROCESSING ASSESSMENT")
        print("=" * 70)
        print("1. Section A - Concept Application")
        print("2. Section B - Task 1 - Text Preprocessor")
        print("3. Section B - Task 2 - TF-IDF Vectoriser")
        print("4. Section B - Task 3 - Naive Bayes Classifier")
        print("5. Section B - Task 4 - Model Comparison")
        print("6. Section C - Mini Capstone Project")
        print("7. Section D - AI-Augmented Learning")
        print("8. Run Section B Tasks 1 to 4")
        print("9. Exit")
        print("=" * 70)

        choice = input("Enter your choice: ")

        if choice == "1":
            section_a()
        elif choice == "2":
            section_b_task1()
        elif choice == "3":
            section_b_task2()
        elif choice == "4":
            section_b_task3()
        elif choice == "5":
            section_b_task4()
        elif choice == "6":
            mini_project()
        elif choice == "7":
            section_d()
        elif choice == "8":
            section_b_task1()
            section_b_task2()
            section_b_task3()
            section_b_task4()
        elif choice == "9":
            print("\nThank you.")
            break
        else:
            print("\nInvalid choice. Please select a valid option.")


if __name__ == "__main__":
    main()
