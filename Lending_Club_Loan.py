#!/usr/bin/env python
# coding: utf-8

# # Predicting Loan Defaults using Deep Learning with Keras & Tensorflow

# # Problem Statement: 
# 
# For companies like Lending Club, predicting loan default with high accuracy is very important. Using the historical Lending Club data from 2007 to 2015, build a deep learning model to predict the chance of default for future loans.

# # Analysis to be done: 
# 
# Perform data preprocessing, exploratory data analysis, and feature engineering. Build a deep learning model to predict load default using the historical public data (https://www.lendingcub.com).

# # Dataset:
# 
# The data set used here can be downloaded from here. The CSV file contains complete loan data for all loans issued through 2007–2015, including the current loan status and payment information. Additional features include annual income, public records, revolving balance, and others.

# # Dataset columns and definition:
# 
#  - credit.policy: “1” — if the customer meets the credit underwriting criteria of LendingClub.com, and “0” otherwise.
#  - purpose: The purpose of the loan (“credit_card”, “debt_consolidation”, “educational”, “major_purchase”, “small_business”, and “all_other”).
#  - int.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be riskier are assigned higher interest rates.
#  - installment: The monthly installments are owed by the borrower if the loan is funded.
#  - log.annual.inc: The natural log of the self-reported annual income of the borrower.
#  - dti: The debt-to-income ratio of the borrower (=amount of debt / annual income).
#  - fico: The FICO credit score of the borrower.
#  - days.with.cr.line: The number of days the borrower has had a credit line.
#  - revol.bal: The borrower’s revolving balance (amount unpaid at the end of the credit card billing cycle).
#  - revol.util: The borrower’s revolving line utilization rate (the amount of the credit line used relative to total credit available).
#  - inq.last.6mths: The borrower’s number of inquiries by creditors in the last 6 months.
#  - delinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
#  - pub.rec: The borrower’s number of derogatory public records (bankruptcy filings, tax liens, or judgments).
# # Let’s jump into the code. I’m breaking code into multiple snippets here. For full code, click here.

# In[ ]:


get_ipython().system('pip install grpcio==1.24.3')
get_ipython().system('pip install tensorflow==2.2.0')
get_ipython().system('pip install pillow')


# In[1]:


import tensorflow as tf
from IPython.display import Markdown, display

def printmd(string):
    display(Markdown('# <span style="color:red">'+string+'</span>'))


if not tf.__version__ == '2.2.0':
    printmd('<<<<<!!!!! ERROR !!!! please upgrade to TensorFlow 2.2.0, or restart your Kernel (Kernel->Restart & Clear Output)>>>>>')


# #  Import libraries

# In[2]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from pickle import dump, load


get_ipython().run_line_magic('matplotlib', 'inline')


# # Read the Loan data CSV and pull the file info.

# In[3]:


df = pd.read_csv('loan_data.csv')


# In[4]:


df.info()
df.head()


# ## The “Purpose” data column is categorical, “Annual income” is log value, which needs to be converted back to exponential. The rest of the columns are numerical. Transpose the data frame to understand the std and mean.

# In[5]:


df.describe().transpose()


# # Check the label “no.fully.paid” distribution in the dataset.

# In[6]:


df['not.fully.paid'].isnull().mean()


# In[7]:


df1=pd.get_dummies(df, columns=['purpose'])


# In[8]:


df1['log.annual.inc'] = np.exp(df1['log.annual.inc'])


# In[9]:


df1.head()


# In[10]:


df.groupby('not.fully.paid')['not.fully.paid'].count()/len(df)


# In[11]:


sns.set_style('darkgrid')
sns.countplot(x='not.fully.paid', data=df)


# The above shows, This dataset is highly imbalanced and includes features that make this problem more challenging. If we do model training with this data, the prediction will be biased since the “not.fully.paid =0 “ has 83.9% filled, and only 16% is the “not.fully.paid=1”

# There were multiple methods to handle imbalanced data; here are a few techniques.
# 1. Resample the training set
# There are two approaches to make a balanced dataset out of an imbalanced one are under-sampling and over-sampling.
# Under-sampling
# Under-sampling balances the dataset by reducing the size of the abundant class. This method is used when the quantity of data is sufficient.
# Over-sampling
# Oversampling is used when the quantity of data is insufficient. It tries to balance the dataset by increasing the size of rare samples.
# There is no absolute advantage of one resampling method over another.
# 2. K-fold Cross-Validation
# Cross-validation should be applied properly while using the over-sampling method to address imbalance problems. Cross-validation should always be done before over-sampling the data.
# If cross-validation is applied after over-sampling, basically, what we are doing is overfitting our model to a specific result.
# 3. Ensemble different resampled datasets
# This approach is simple and perfectly horizontally scalable if you have a lot of data since you can train and run your models on different cluster nodes. Ensemble models also tend to generalize better, which makes this approach easy to handle.
# 4. Resample with different ratios
# The previous approach can be fine-tuned by playing with the ratio between the rare and the abundant class. The best ratio heavily depends on the data and the models that are used. But instead of training all models with the same ratio in the ensemble, it is worth trying to ensemble different ratios. So if 10 models are trained, it might make sense to have a model that has a ratio of 1:1 (rare:abundant) and another one with 1:3, or even 2:1. Depending on the model used, this can influence the weight that one class gets.
# 5. Cluster the abundant class
# An elegant approach was proposed by Sergey on Quora [2]. Instead of relying on random samples to cover the variety of the training samples, he suggests clustering the abundant class in r groups, with r being the number of cases in r. For each group, only the medoid (center of cluster) is kept. The model is then trained with the rare class and the medoids only.
# 6. Design your own models
# All the previous methods focus on the data and keep the models as a fixed component. But in fact, there is no need to resample the data if the model is suited for imbalanced data. The famous XGBoost is already a good starting point if the classes are not skewed too much because it internally takes care that the bags it trains on are not imbalanced. But then again, the data is resampled; it is just happening secretly.
# By designing a cost function that is penalizing the wrong classification of the rare class more than wrong classifications of the abundant class, it is possible to design many models that naturally generalize in favor of the rare class. For example, tweaking an SVM to penalize wrong classifications of the rare class by the same ratio that this class is underrepresented.

# # The dataset used here is minimal; I chose to try oversampling to balance this dataset.

# In[12]:


count_class_0, count_class_1 = df['not.fully.paid'].value_counts()


# In[13]:


df_0 = df[df['not.fully.paid'] == 0]
df_1 = df[df['not.fully.paid'] == 1]


# In[14]:


df_1_over = df_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_0, df_1_over], axis=0)


# In[15]:


print('Random over-sampling:')
print(df_test_over['not.fully.paid'].value_counts())


# In[16]:


#df_test_over['not.fully.paid'].value_counts().plot(kind='bar', title='Count (not.fully.paid)')

sns.set_style('darkgrid')
sns.countplot(x='not.fully.paid', data=df_test_over)


# # Exploratory Data Analysis
# Let’s see some data visualization with seaborn and plotting. Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.

# In[17]:


df['revol.bal'].hist(figsize=[12,6], bins=50)


# In[18]:


df1=pd.get_dummies(df, columns=['purpose'])


# In[19]:


plt.figure(figsize=(10,6))
df[df['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',bins=30,label='Credit.Policy=1')
df[df['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# # Let's see a similar chart for “not.fully.paid” column.
# 

# In[20]:


plt.figure(figsize=(10,6))
df[df['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
df[df['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# Now, check the dataset group by loan purpose. Create a countplot with the color hue defined by not.fully.paid.

# In[21]:


plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=df,palette='Set1')


# The next visual we will pull part of EDA in this dataset is the trend between FICO score and interest rate.

# In[22]:


sns.jointplot(x='fico',y='int.rate',data=df,color='purple')


# To compare the trend between not.fully.paid and credit.policy, create seaborn implot.

# In[23]:


plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=df,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# The above visuals gave us an idea of how the data is and what we will work with. Nest step is to prepare the data for model training and test as the first step converts the categorical values to numeric. Here in this dataset “purpose” column is a critical data point for the model as per our analysis above, and it is categorical.

# In[24]:


cat_feats = ['purpose']
#cat_feats =df_test_over


# In[25]:


#final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)
final_data = pd.get_dummies(df_test_over,columns=cat_feats,drop_first=True)


# In[26]:


final_data.info()
final_data.head()


# In[27]:


final_data.corr()


# In[28]:


final_data.corr()
plt.figure(
        figsize=[16,12]
)

sns.heatmap(
        data=final_data.corr(), 
        cmap='viridis', 
        annot=False, 
        fmt='.2g'
)


# We only focus on the grids of yellow or very light green. After comparing with the feature description again, I decided to drop:’revol.bal’, ‘days.with.cr.line’, ‘installment’, ‘revol.bal’
# 
# revol.bal, day.with.cr.line, installment can represent by annual income. revol.util can represent by int.rate.

# # Modeling
# 
# # Deep Learning Implementation
# 
# Finally, do the train test split and fit the model with the data shape we created above. since there are 19 features, I chose the first layer of the neural network with 19 nodes.

# In[29]:


to_drop2 = ['revol.bal', 'days.with.cr.line', 'installment', 'revol.bal']

final_data.drop(to_drop2, axis=1, inplace=True)
#We only focus on the grids of yellow or very light green. After comparing with the feature description again,  revol.bal,day.with.cr.line,installment c


# In[30]:


final_data.isnull().mean()


# In[31]:


#to_train = df1[df1['not.fully.paid'].isin([0,1])]
#to_pred = df1[df1['not.fully.paid'] == 2]

to_train = final_data[final_data['not.fully.paid'].isin([0,1])]
to_pred = final_data[final_data['not.fully.paid'] == 2]


# In[32]:


X = to_train.drop('not.fully.paid', axis=1).values
y = to_train['not.fully.paid'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 101)


# In[33]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[34]:


X_train.shape


# In[35]:


model = Sequential()

model.add(
        Dense(94, activation='relu')
)

model.add(
        Dense(30, activation='relu')
)

model.add(
        Dense(15, activation='relu')
)


model.add(
        Dense(1, activation='sigmoid')
)

model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
)


# In[36]:


early_stop = EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        verbose=1, 
        patience=25
)

model.fit(
        X_train, 
        y_train, 
        epochs=200, 
        batch_size=256, 
        validation_data=(X_test, y_test),
         callbacks=[early_stop]
)


# # Model Evaluation and Validation

# In[37]:


pd.DataFrame(model.history.history)[['loss','val_loss']].plot() #over fitting 


# # This validation result, the Loss plot, shows us the model is overfitted.

# In[38]:


predictions = model.predict_classes(X_test)

print(
        confusion_matrix(y_test,predictions), 
        '\n', 
        classification_report(y_test,predictions)
)


# # Classification report
# 
# The model’s overall f1-score for accuracy is 0.69. Still, there are type 2 errors (624) in the prediction.

# # Model Refinement
# 
# Two ways of refining the model we will try here. Add Dropout layers to bring down the overfitting OR Lower the cut-off line in binary prediction to reduce the Type 2 error, at the cost of increasing Type 1 error. In the LendingClub case, Type 2 error is the more serious problem because it devastates its balance sheet, while Type 1 error is not a very big deal.

# In[39]:


model_new = Sequential()

model_new.add(
        Dense(94, activation='relu')
)

model_new.add(Dropout(0.2))

model_new.add(
        Dense(30, activation='relu')
)

model_new.add(Dropout(0.2))

model_new.add(
        Dense(15, activation='relu')
)

model_new.add(Dropout(0.2))

model_new.add(
        Dense(1, activation='sigmoid')
)

model_new.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['binary_accuracy']
)


model_new.fit(
        X_train, 
        y_train, 
        epochs=200, 
        batch_size=256, 
        validation_data=(X_test, y_test),
         callbacks=[early_stop]
)


# In[40]:


pd.DataFrame(model_new.history.history)[['loss','val_loss']].plot() #The graph shows that, by adding in Dropout layers, we have reduced the overfitting issue compared with the old model


# The graph shows that, by adding in Dropout layers, we have reduced the overfitting issue compared with the old model.

# In[41]:


predictions_new = (model_new.predict_proba(X_test) >= 0.2).astype('int')

print(
        confusion_matrix(y_test,predictions_new), 
        '\n', 
        classification_report(y_test,predictions_new)
)


# By changing the cut-off line to 0.2 (default is 0.5), we have dramatically brought down the Type 2 error.

# # Save the model and scalar.

# In[42]:


dump(scaler, open('scaler.pkl', 'wb'))
model_new.save('my_model_lending_club.h5')


# # Model Use Case
# 
# We will use the model on “not.fully.paid = 0” records; when these loans are matured, we will get it as the Out-Of-Time sample validation results.
# In the future, this model can be used on any new customer to provide some insight when deciding whether to grant the loan.

# In[43]:


later_scaler = load(open('scaler.pkl', 'rb'))
later_model = load_model('my_model_lending_club.h5')


# In[53]:


X_OOT = to_pred.drop('not.fully.paid', axis=1).values

print(X_OOT.shape)


# # Conclusion
# 
# When building the Neural Network, the most difficult part is the Sequential Model because there are many different options available in building the layers. The way how to come up with the optimized number of layers and nodes are remaining challenging.

# # Credits:
# 
# Github repository “Capstone-Lending-Club” by Sean329 Kaggle notebook “Lending Club Loan Analysis” by renjitmishra 7 Techniques to Handle Imbalanced Data By Ye Wu & Rick Radewagen, IE Business School. 
# Full code link: https://github.com/sarathi-tech/lending-club/blob/main/Lending_Club_colab.ipynb
# 
