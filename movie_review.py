
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
from TextCNN import TextCNN
import gzip
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from Embedder import Embedder


# In[2]:


FILENAME = "movies.txt"
COLUMNS = ["productId",
           "userId",
           "profileName",
           "helpfulness",
           "score",
           "time",
           "summary",
           "text"]
COL_NUM = 8
REVIEW_NUM = 1000
PADDED_LEN = 200
GLOVE_BINARY_PATH = "glove.6B.50d.txt"


# In[3]:


df = pd.read_csv(FILENAME,
                 header=None,
                 delimiter="\n",
                 error_bad_lines=False,
                 skip_blank_lines=True,
                 encoding="latin-1",
                 nrows=COL_NUM*REVIEW_NUM)
df = pd.DataFrame(np.reshape(df.values,(REVIEW_NUM,COL_NUM)),columns=COLUMNS)


# In[4]:


df["productId"] = df["productId"].str.replace("product/productId:","")
for col in COLUMNS[1:]:
    df[col] = df[col].str.replace("review/" + col + ":","")


# In[5]:


X = df["text"]
y = df["score"].astype("float").astype("int").values.reshape([-1,1])
y_one_hot = OneHotEncoder().fit_transform(y).toarray()


# In[6]:


embedder = Embedder(None,50,PADDED_LEN,GLOVE_BINARY_PATH)
X_embedded = embedder.str_series_to_image(X)


# In[7]:


import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader


# In[8]:


LEARNING_RATE = 0.0025
FC_LAYER = 30
CLASSES_LEN = 5
EMBEDDING_DIM = 50
CONV_FILTERS = 15
EPOCHS = 100
TRAIN_TEST_RATION = 0.2


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X_embedded, y_one_hot, test_size=TRAIN_TEST_RATION)
X_train = X_train.reshape([-1,1,EMBEDDING_DIM,PADDED_LEN])
X_test = X_test.reshape([-1,1,EMBEDDING_DIM,PADDED_LEN])
X_train_tensor = torch.Tensor(X_train)
X_test_tensor = torch.Tensor(X_test)
y_train_tensor = torch.Tensor(y_train)
y_test_tensor = torch.Tensor(y_test)
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
test_dataset = TensorDataset(X_test_tensor,y_test_tensor)
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[24]:


X_test.shape


# In[25]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[26]:


model = TextCNN(hidden_units=FC_LAYER,
                output_len=CLASSES_LEN,
                textcnn_filter_count=CONV_FILTERS,
                sentence_max_size=PADDED_LEN,
                word_embedding_dimension=EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
model.to(device)


# In[27]:


for epoch in range(EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

