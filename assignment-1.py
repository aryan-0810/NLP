def question_1():
    print("\n" + "=" * 80)
    print("1. WHAT IS NATURAL LANGUAGE PROCESSING (NLP)")
    print("=" * 80)

    answer = """
Natural Language Processing (NLP) is a branch of Artificial Intelligence that
helps computers understand, process, and work with human language. It allows
machines to read text, understand user messages, identify the meaning of words,
and respond to users. We use NLP in many applications in our daily life. For
example, WhatsApp can use NLP to understand messages for features such as
smart replies, message suggestions, and chat-based interactions.
"""

    print(answer)


def question_2():
    print("\n" + "=" * 80)
    print("2. IMPORTANCE OF NLP IN AI APPLICATIONS")
    print("=" * 80)

    points = [
        (
            "1. Understanding User Queries",
            "NLP helps chatbots understand what users are asking. "
            "For example, a Swiggy chatbot can understand a message such as "
            "'I want to order a pizza' and provide relevant options."
        ),
        (
            "2. Spam Detection",
            "NLP helps email applications identify unwanted or suspicious "
            "messages. For example, Gmail can analyze the words and meaning "
            "of an email and move spam messages to the spam folder."
        ),
        (
            "3. Sentiment Analysis",
            "NLP can identify whether a user's review or message expresses "
            "a positive, negative, or neutral opinion. This helps applications "
            "understand customer feedback and improve the user experience."
        )
    ]

    for title, description in points:
        print(f"\n{title}")
        print(description)


def question_3():
    print("\n" + "=" * 80)
    print("3. TOKENIZATION USING split()")
    print("=" * 80)

    sentence = "I love ordering pizza from Zomato!"
    tokens = sentence.split()

    print("Sentence:")
    print(sentence)

    print("\nTokens:")
    print(tokens)


def question_4():
    print("\n" + "=" * 80)
    print("4. SIMPLE SENTIMENT CHECKER")
    print("=" * 80)

    review = input("Enter your movie review: ").lower()

    words = review.split()

    if "good" in words:
        print("Positive")
    elif "bad" in words:
        print("Negative")
    else:
        print("Neutral")


def is_spam(email_text):
    email_text = email_text.lower()
    words = email_text.split()

    if "win" in words or "free" in words:
        return True

    return False


def question_5():
    print("\n" + "=" * 80)
    print("5. SIMPLE SPAM FILTER")
    print("=" * 80)

    email_1 = "Congratulations! You win a free gift."
    email_2 = "Your meeting is scheduled for tomorrow."

    print("Email 1:")
    print(email_1)
    print("Is Spam:", is_spam(email_1))

    print("\nEmail 2:")
    print(email_2)
    print("Is Spam:", is_spam(email_2))


def main():
    print("=" * 80)
    print("NATURAL LANGUAGE PROCESSING ASSIGNMENT")
    print("=" * 80)
    print("Name: Aryan Pandya")
    print("Subject: Natural Language Processing")
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