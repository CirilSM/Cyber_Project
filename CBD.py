#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
pd.options.mode.chained_assignment = None

#import requests as rq
import numpy as np
import re

#!pip install wordninja
import wordninja

#!pip install spacy
#import spacy
#spacy.cli.download("en_core_web_sm")

import nltk
from nltk.tokenize import word_tokenize
#nltk.download('punkt')

#!pip install contractions
import contractions

#!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score   
from sklearn.metrics import f1_score

import mysql.connector

import pickle

mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "my123SQL$",
    database = "modeldata"
)

command = mydb.cursor()

class CyberbullyingDetection():
    def string(self, data):
        string = ' '
        return (string.join(data))

    def detectBullying(self, text):
        tweet = pd.DataFrame(text, columns=['tweets_text'])
        tweet = tweet.replace(to_replace= '\\r', value= '', regex=True)
        
        tweet['tweets_text'] = tweet['tweets_text'].str.lower()
        
        temp = ''
        for index, row in enumerate(tweet['tweets_text']):
            temp = re.sub(r'(\brt)|(http\S+)|(\d+)|(&(gt;)+)|(&(lt;)+)|(&(amp;)+)|([^\w\s])', '', str(row))
            temp = re.sub('(\'| )|(\"| )|(_)', ' ', temp)
            tweet['tweets_text'][index] = temp
        
        for index, row in enumerate(tweet['tweets_text']):
            temp = []
            for word in row.split():
                temp.append(contractions.fix(word))
            tweet['tweets_text'][index] = self.string(temp)         
            
        command.execute("SELECT * FROM slang")
        slangWords = pd.DataFrame(command, columns=['slang', 'word'])
        slangWords = slangWords.replace(to_replace= '\\r', value= '', regex=True)    
        
        for num, row in enumerate(tweet['tweets_text']):
            temp = []
            for word in row.split():
                found = 0
                if (len(word)<6 and len(word)>2): 
                    for index, slang in enumerate(slangWords['slang']):
                        if (slang == word):
                            temp.append(slangWords['word'][index])
                            found = 1
                if (found != 1):
                    temp.append(word)
            tweet['tweets_text'][num] = self.string(temp)
            
        for index, row in enumerate(tweet['tweets_text']):
            temp = []
            for word in row.split():
                if (len(word)>8):
                    unmunched = wordninja.split(word)
                    temp.append(self.string(unmunched))
                else:
                    temp.append(word)
            tweet['tweets_text'][index] = self.string(temp)
            
        tokens = []
        for row in tweet['tweets_text']:
            tokens.append(word_tokenize(row))
        
        tweet['tokens'] = tokens   
        
        command.execute("SELECT * FROM offensivewithseverity")
        offenseWords = pd.DataFrame(command, columns=['word', 'severity'])
        offenseWords = offenseWords.replace(to_replace= '\\r', value= '', regex=True)
        
        command.execute("SELECT * FROM negation")
        negationWords = pd.DataFrame(command, columns=['word'])
        negationWords = negationWords.replace(to_replace= '\\r', value= '', regex=True)

        totalWords, offensiveWords, severityWords = [], [], []

        for row in tweet['tokens']:
            words, temp1, temp2 = 0, [], []
            for index1, token in enumerate(row):
                words += 1
                for index2, offensive in enumerate(offenseWords['word']):
                    if (token == offensive):
                        negation = 0
                        for negation in negationWords['word']: #Checking for negation words at most 2 words before the negative word 
                            if (index1<1):
                                break
                            if (row[index1-1] == negation or row[index1-2] == negation):
                                negation = 1
                                break
                        if (negation != 1):
                            temp1.append(token)
                            temp2.append(offenseWords['severity'][index2])
            totalWords.append(words)
            offensiveWords.append(temp1)
            severityWords.append(temp2)

        tweet['total words'] = totalWords
        tweet['offensive words'] = offensiveWords
        tweet['severity words'] = severityWords
        
        density = []
        for total, offensive in zip(tweet['total words'], tweet['offensive words']):
            density.append(len(offensive) / total)
        tweet['density'] = density
        
        compound = []
        for row in tweet['tweets_text']:
            polarity = SentimentIntensityAnalyzer().polarity_scores(row)
            compound.append(polarity["compound"])
        tweet['sentiment analysis'] = compound
        
        severity, weights = [], [1, 2, 3, 4, 5]
        for severe in tweet['severity words']:
            count, product = [0] * 5, []
            for num in severe:
                if (num == '1'):
                    count[0] += 1
                elif (num == '2'):
                    count[1] += 1 
                elif (num == '3'):
                    count[2] += 1
                elif (num == '4'):
                    count[3] += 1 
                elif (num == '5'):
                    count[4] += 1       
            for num1, num2 in zip(count, weights):
                product.append(num1 * num2)

            totalProduct = sum(product)
            totalCount = sum(count)

            if (totalCount == 0):
                severity.append(0)
            else:
                severity.append(totalProduct / totalCount)

        tweet['severity'] = severity
        
        tweetDataM1 = tweet[['density', 'severity', 'sentiment analysis']].copy()
        tweetDataM1.head()
        
        model = pickle.load(open("cyberbullyingdetection.sav", 'rb'))
        
        cyberTarget = model.predict(tweetDataM1)
        if self.string(cyberTarget) == 'cyberbullying':
            tweet['cyberbullying'] = 'True'
        else:
            tweet['cyberbullying'] = 'False'
            
        if (tweet['cyberbullying'].values == 'True'):   
            command.execute("SELECT * FROM ethnicityAndRaceGlossary")
            ethnicityAndRaceGlossary = pd.DataFrame(command, columns=['word'])
            ethnicityAndRaceGlossary = ethnicityAndRaceGlossary.replace(to_replace= '\\r', value= '', regex=True)
            ethnicityAndRaceGlossary.head()

            command.execute("SELECT * FROM ageGlossary")
            ageDataGlossary = pd.DataFrame(command, columns=['word'])
            ageDataGlossary = ageDataGlossary.replace(to_replace= '\\r', value= '', regex=True)
            ageDataGlossary.head()

            command.execute("SELECT * FROM genderGlossary")
            genderDataGlossary = pd.DataFrame(command, columns=['word'])
            genderDataGlossary = genderDataGlossary.replace(to_replace= '\\r', value= '', regex=True)
            genderDataGlossary.head()

            command.execute("SELECT * FROM religionGlossary")
            religiousDataGlossary = pd.DataFrame(command, columns=['word'])
            religiousDataGlossary = religiousDataGlossary.replace(to_replace= '\\r', value= '', regex=True)
            religiousDataGlossary.head()

            isEthnicityAndRace = []
            for row in tweet['tokens']:
                temp = 0
                for token in row:
                    for glossary in ethnicityAndRaceGlossary['word']:
                        if (token == glossary):
                            temp += 1
                            break
                isEthnicityAndRace.append(temp)

            tweet['ethnicity and race'] = isEthnicityAndRace

            isAge = []
            for row in tweet['tokens']:
                temp = 0
                for token in row:
                    for glossary in ageDataGlossary['word']:
                        if (token == glossary):
                            temp += 1
                            break
                isAge.append(temp)

            tweet['age'] = isAge

            isGender = []
            for row in tweet['tokens']:
                temp = 0
                for token in row:
                    for glossary in genderDataGlossary['word']:
                        if (token == glossary):
                            temp += 1
                            break
                isGender.append(temp)

            tweet['gender'] = isGender  

            isReligious = []
            for row in tweet['tokens']:
                temp = 0
                for token in row:
                    for glossary in religiousDataGlossary['word']:
                        if (token == glossary):
                            temp += 1
                            break
                isReligious.append(temp)

            tweet['religion'] = isReligious

            tweetDataM2 = tweet[['age', 'gender', 'religion', 'ethnicity and race']].copy()

            model = pickle.load(open("cyberbullyingtype.sav", 'rb'))

            classifyTarget = model.predict(tweetDataM2)

            print('\nOffensive Words: ', tweet['offensive words'][0], '\nSeverity Level: ', tweet['severity'][0], '\nType: ', classifyTarget)
    
        else:
            print("No offensive words!")

text = ["You're A retardHISPANIC, all you do is drink tequilla and mow lawns you weirdo beaner!"];
scan = CyberbullyingDetection()
scan.detectBullying(text)

