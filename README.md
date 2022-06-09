# Wine-Reviews (NLP)

# Building three Classification models; Logistic Regression, Decision Tree, and Random Forest based on wine descriptions

By:Ayma Rahman, Sarah Saleem, Samira Shafiei, Kate Zagrebneva

# Objectives
Find insights using Exploratory Data Analysis (e.g. Wordcloud)
and Building three Classification models; Logistic Regression, Decision Tree, and Random Forest based on wine descriptions to predict whether the wine is:
   - Popular wine or not popular wine?
   - Cheap wine or expensive wine?
   - White wine or red wine

# Data 
We used dataset from Kaggle called “winemag”.  Originally, The data was scraped from WineEnthusiast during the week of June 15th, 2017. There are 13 attributes and 129970 observations in the dataset.

# Data Preprocessing
Mainly six preliminary data processing steps were undertaken for cleaning and correcting the data for the purpose of analysis:

**1. Removing unnecessary columns**: For this project we only needed a few variables such as Description, country, points, and price. We removed the rest of the variables.

**2. Dealing with Missing Values, duplicates, and outliers**: There were no duplicates in the dataset, however, there were 204752 missing values. We dropped the NAs in the variables that were used in our models.There were some outliers in the dataset. Since most machine learning algorithms do not work well in the presence of outliers, we decided to drop them. For example, in price, there were a few outliers over $900 which had a significant effect on the accuracy. After removing them we could get higher accuracy.

**3. Tokenization**: We are working with text, and the most important step in text preprocessing is tokenization to transform the reviews into vectors. We tried two methods such as Countvectorizer and TF-IDF, and CountVectorizer gave us higher accuracies. we used it to transform the reviews into vectors. 

**4. Lower casing, removing punctuations, and removing Stop words**: Also, we changed all the words to lower case, removed the punctuations, and also removed the stop words in the description column.

**5. Create different classes for the target variables**: For building our models, we had to create 2-3 classes for each target variable:
  -Popularity: Good, Very Good, Excellent
  -Price: we created two classes for Price; Cheap Wines and Expensive Wines 
  -Type: Red, White

**6. Label Encoding**: We used label encoding to assign unique integers to each class in each variable to work better in our models, especially in the Logistic Regression model.

# Project Analysis
We processed the wine description text to create three classification models, such as Logistic Regression, Decision Tree, and Random Forest to classify the wines into popularity, price, and type. By having the top word sequences we can have insight into the language describing wines. To do so, we will be using n-grams, Countvectorizer, word cloud, and nltk (Natural Language Toolkit) python packages. We will use unigram, bigrams, and trigrams to find the top one word, two sequence words, and three sequence words to describe wine.

**Popularity:** We used the description column, which is the tasters' reviews and Points column as the target variabel. After trying different classes for the points to get the highest accuracy, we realized having three categories: good, very good, and excellent give the best accuracy. Based on the accuracy of all three models, we see that logistic and random forest performed the best with the latter being a little bit higher. For the quality prediction, it is suitable to use a random forest model as we got the highest accuracy when compared to the other two.

**Price:** We built three classification models to predict whether the wine is cheap or expensive. We used the description column, which is the tasters' reviews, and the price column as the target variable.  We tried many grouping options, and after running logistic regression, decision tree, and random forest models, we found out that having two groups (cheap and expensive) gives us the highest accuracy in all three model. Random forest performs slightly better than Logistic Regression in terms of accuracy, but Logistic regression is better in terms of Recall. Based on the recall comparison and accuracy, the best model to classify whether the wine is cheap or expensive would be Logistic Regression.

**Wine Color (Type):** The dataset had a total of 707 unique wine varieties. We selected the top 16% of the data which boiled down to 113 different wines. And we used that sample to categorize them into red and white. However, we were not provided those labels in our dataset so we first had to manually create them. We did some research on red and white wines and were able to figure out certain words that would indicate the wine color. So we built a list of those terms and wrote a simple code that would replace the “variety” description with the corresponding color if one of the key words were present in the column. The red and white wines had a total of 53,335 and 64,430 rows respectively leaving only about 2000 rows classified as “Other” which we later dropped before running our models.For this classification, we see that all three models are pretty close to each other in terms of their accuracy. However random forest performed the best with an accuracy of 88.9%. When comparing the f1 and recall scores, we see that the recall for white wines is better in Logistic Regression and Random Forest than Decision Trees. For red wine, the recall is consistent, however the f1 score is the highest in Random Forest.

We have a class imbalance in price, so we tried different machine learning techniques such as oversampling and undersampling to address this issue. After oversampling, the accuracy in all three models decreased. In undersampling, we got very similar scores for the imbalance dataset. 


