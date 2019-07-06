# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
#from .BasicModule import BasicModule
from torch.nn.modules import Module


class TextCNN(Module):

    def __init__(self,output_len,word_embedding_dimension,sentence_max_size,hidden_units,textcnn_filter_count):
        super(TextCNN, self).__init__()
        self.out_channel = output_len
        self.label_num = output_len
        self.conv3 = nn.Conv2d(1, textcnn_filter_count, ( word_embedding_dimension,3))
        self.conv4 = nn.Conv2d(1, textcnn_filter_count, ( word_embedding_dimension,4))
        self.conv5 = nn.Conv2d(1, textcnn_filter_count, ( word_embedding_dimension,5))
        self.conv6 = nn.Conv2d(1, textcnn_filter_count, ( word_embedding_dimension,8))
        self.conv7 = nn.Conv2d(1, textcnn_filter_count, ( word_embedding_dimension,11))
        self.conv8 = nn.Conv2d(1, textcnn_filter_count, ( word_embedding_dimension,20))
        
        self.Avg3_pool = nn.AvgPool2d((1,sentence_max_size-3+1))
        self.Avg4_pool = nn.AvgPool2d((1,sentence_max_size-4+1))
        self.Avg5_pool = nn.AvgPool2d((1,sentence_max_size-5+1))
        self.Avg6_pool = nn.AvgPool2d((1,sentence_max_size-8+1))
        self.Avg7_pool = nn.AvgPool2d((1,sentence_max_size-11+1))
        self.Avg8_pool = nn.AvgPool2d((1,sentence_max_size-21+1))

        self.Max3_pool = nn.MaxPool2d((1,sentence_max_size-3+1))
        self.Max4_pool = nn.MaxPool2d((1,sentence_max_size-4+1))
        self.Max5_pool = nn.MaxPool2d((1,sentence_max_size-5+1))
        self.Max6_pool = nn.MaxPool2d((1,sentence_max_size-8+1))
        self.Max7_pool = nn.MaxPool2d((1,sentence_max_size-11+1))
        self.Max8_pool = nn.MaxPool2d((1,sentence_max_size-21+1))
        
        self.fc1 = nn.Linear(12*textcnn_filter_count,hidden_units)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_units)
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(hidden_units, hidden_units // 2)
        self.bn2 = nn.BatchNorm1d(hidden_units // 2)
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(hidden_units // 2, hidden_units // 4)
        self.bn3 = nn.BatchNorm1d(hidden_units // 4)
        self.dropout3 = nn.Dropout(p=0.3)
        
        self.linear1 = nn.Linear(hidden_units // 4, output_len)

        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.conv5.weight)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.linear1.weight)

    def forward(self, x):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        x = x.type(torch.cuda.FloatTensor)
        batch = x.shape[0]

        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))
        x4 = F.relu(self.conv6(x))
        x5 = F.relu(self.conv7(x))
        x6 = F.relu(self.conv8(x))
        # Pooling
        x1_max = self.Max3_pool(x1)
        x2_max = self.Max4_pool(x2)
        x3_max = self.Max5_pool(x3)
        x4_max = self.Max6_pool(x4)
        x5_max = self.Max7_pool(x5)
        x6_max = self.Max8_pool(x6)

        x1_avg = self.Avg3_pool(x1)
        x2_avg = self.Avg4_pool(x2)
        x3_avg = self.Avg5_pool(x3)
        x4_avg = self.Avg6_pool(x4)
        x5_avg = self.Avg7_pool(x5)
        x6_avg = self.Avg8_pool(x6)

        x1_avg = x1_avg.view(batch, -1)
        x2_avg = x2_avg.view(batch, -1)
        x3_avg = x3_avg.view(batch, -1)
        x4_avg = x4_avg.view(batch, -1)
        x5_avg = x5_avg.view(batch, -1)
        x6_avg = x6_avg.view(batch, -1)

        x1_max = x1_max.view(batch, -1)
        x2_max = x2_max.view(batch, -1)
        x3_max = x3_max.view(batch, -1)
        x4_max = x4_max.view(batch, -1)
        x5_max = x5_max.view(batch, -1)
        x6_max = x6_max.view(batch, -1)
        # capture and concatenate the features
        x = torch.cat((x1_avg, x2_avg, x3_avg,x4_avg,x5_avg,x6_avg,x1_max, x2_max, x3_max,x4_max,x5_max,x6_max),-1)
        x = x.view(batch, -1)

        # project the features to the labels
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.dropout(self.dropout1(x))
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.dropout(self.dropout2(x))
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.dropout(self.dropout3(x))
        x = F.log_softmax(self.linear1(x))


        return x
