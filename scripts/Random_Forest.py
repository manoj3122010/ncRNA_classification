#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('/home/myoui/shared/ML/gene_list_para.csv')


# In[3]:


label_encoder = LabelEncoder()
df['gene_type_encoded'] = label_encoder.fit_transform(df['gene_type'])
y = df['gene_type_encoded']


# In[4]:


scaler = MinMaxScaler()
df['mfe_scaled'] = scaler.fit_transform(df[['mfe']])


# In[5]:


df


# In[6]:


# Feature extraction: Select relevant columns and k-mer text data
X_numerical = df[['gc_content', 'seq_length', 'dot_count', 'bracket_count', 'mfe_scaled']]
kmers_text = df['kmers']


# In[7]:


# Vectorize k-mers using CountVectorizer
vectorizer = CountVectorizer()
X_kmers = vectorizer.fit_transform(kmers_text)


# In[8]:


# Convert k-mer matrix to a DataFrame and reset index to merge with X_numerical
X_kmers_df = pd.DataFrame(X_kmers.toarray(), columns=vectorizer.get_feature_names_out())
X_kmers_df.reset_index(drop=True, inplace=True)


# In[9]:


X = pd.concat([X_numerical.reset_index(drop=True), X_kmers_df], axis=1)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[12]:


y_pred = clf.predict(X_test)


# In[13]:


print(classification_report(y_test, y_pred))


# In[14]:


conf_matrix = confusion_matrix(y_test, y_pred)


# In[ ]:


plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# In[ ]:




