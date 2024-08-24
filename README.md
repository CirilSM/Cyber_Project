# **SafeNet**

## **Goal**
The goal of this project is to predict whether or not a particular text can be classed as hostile by utilizing machine learning and natural language processing (NLP). Additionally, try to pinpoint the precise kind of objectionable content—for example, content that offends based on someone's age, gender, religion, or race. By combining cutting-edge NLP and ML approaches, the method analyzes text data and extracts relevant patterns. We have in the project two different labeled datasets. One identifies whether a text has cyberbullying, and the second identifies the type, such as race, religion, gender, and age. 

## **Dataset**
The message is passed through different phases. First of all, various data preprocessing techniques are applied to it, including lowercasing, regular expression-based cleaning, expansion of contractions and slangs,
and handling of munched words. Various attributes are extracted from it, including density of offensive words, sentimental analysis, and weighted average of offensive words. For prediction of cyberbullying and its specific type, different classification models are used that include SVC, Logistic Regression, Random Forest Classifier, KNN, Naive Bayes and Extra Trees Classifier. It returns results that the Random Forest Classifier holds an accuracy of 85% in identifying cyberbullying, and logistic regression turns out to be 94% accurate in classifying the type of cyberbullying.

## **Objectives**
•	A project aimed at correct detection of cyberbullying and type of bullying is presence of text.
•	It will help in making tailored responses and effective moderation and get insights about
•	particular type with detection of bullying type like race, religion, age, and gender.
•	Analyze textual data for detection of cyberbullying from it through linguistic patterns using NLP.
•	Apply machine learning models for predicting whether a text can be classified as offensive or safe.
•	Make a positive project to remove online hate and abuses from the internet so that the user is free from any harm or abusive selling.
•	Our project keeps on evolving hand in hand to cope up with the new patterns of abusive language and provide the best service to our client.

## **Data Collection**
For this project, We used a Kaggle dataset containing over 47,000 tweets classified by cyberbullying types (age, ethnicity/race, gender, religion, others) and non-cyberbullying. To simplify analysis, we reduced the dataset to approximately 250 instances per class. The project is divided into two parts: first, identifying whether a tweet contains cyberbullying, and second, classifying the type of cyberbullying.  We compiled wordlists related to age, race, gender, religion, offensive terms with severity, slang abbreviations, and negations, all stored in a database as separate tables.

## **Data Preprocessing**
We'll preprocess the data by converting the dataset to lowercase for consistent wordlist matching. Next, we'll clean the data using regular expressions to remove @mentions, digits, "RT", links, and special characters. Contractions (e.g., "aren’t" to "are not") and slangs (e.g., "hml" to "hate my life") will be expanded for accuracy. We'll use the WordNinja library to split combined words (e.g., "hatemylife" to "hate my life") for better slang detection. Finally, we'll tokenize the text to prepare for further processing.

## **Data Transformation**
To prepare data for our model, we'll convert text into numerical features. First, we'll calculate the total word count and offensive word count to create the "density of offensive words" feature. Next, we'll perform sentiment analysis to generate a polarity feature. Then, we'll calculate the severity of offensive words using a weighted average as the final feature for detecting cyberbullying. If cyberbullying is detected, the model will extract four additional features—word counts related to age, race, religion, and ethnicity—using relevant glossaries to determine the specific type of bullying.

## **Model Development**
We'll create our model by splitting the dataset into 85% training and 15% testing data, ensuring a balanced split. We'll then train and evaluate six different models for comparison: SVC, Logistic Regression, Random Forest, KNN, Naïve Bayes, and Extra Tree Classifier.

## **Model Evaluation**
![image](https://github.com/user-attachments/assets/f5ad49e0-5350-4bc6-9185-99fc0379b76c)

## **Conclusion**
Through data preprocessing, feature extraction, and classification modeling, we developed a data science project to identify and potentially eliminate cyberbullying. Using Python, Jupyter Notebook, and various libraries, we processed textual data for model implementation. Our models excelled in detecting and classifying cyberbullying with high accuracy. The project's mission is to create a positive, safe environment by eliminating the negativity and toxicity of cyberbullying.


