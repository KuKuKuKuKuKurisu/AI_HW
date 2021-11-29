
import numpy as np
from collections import Counter
from torch import nn
from torch.nn import Sequential
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch

def onehot_matrix(samples_vec, num_classes):
    """
    >>> onehot_matrix(np.array([1, 0, 3]), 4)
    [[ 0.  1.  0.  0.]
     [ 1.  0.  0.  0.]
     [ 0.  0.  0.  1.]]

    >>> onehot_matrix(np.array([2, 2, 0]), 3)
    [[ 0.  0.  1.]
     [ 0.  0.  1.]
     [ 1.  0.  0.]]

    Ref: http://bit.ly/1VSKbuc
    """
    num_samples = samples_vec.shape[0]

    onehot = np.zeros(shape=(num_samples, num_classes))
    onehot[range(0, num_samples), samples_vec] = 1

    return onehot

def segment_based_evaluation(y_pred, segment_ids, segment2label):
    """
    @argments:
    y_pred: predicted labels of frames
    segment_ids: segment id of frames
    segment2label: mapping from segment id to label
    """
    seg_pred = {}
    for frame_id, seg_id in enumerate(segment_ids):
        if seg_id not in seg_pred:
            seg_pred[seg_id] = []
        seg_pred[seg_id].append(y_pred[frame_id])

    ncorrect = 0
    for seg_id in seg_pred.keys():
        predicted = seg_pred[seg_id]
        c = Counter(predicted)
        predicted_label = c.most_common()[0][0] # take the majority voting

        if predicted_label == segment2label[seg_id]:
            ncorrect += 1

    accuracy = ncorrect/len(segment2label)
    print('Segment-based Accuracy using %d testing samples: %f' % (len(segment2label), accuracy))

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.lstm=nn.LSTM(input_size=3,hidden_size=64,batch_first=True)
        self.fcmodel=Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=64,out_features=64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=64,out_features=32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=32,out_features=7)
        )
    def forward(self,x):
        x,(hn,cn)=self.lstm(x)
        x=x.permute([1,0,2])
        x=x[-1]
        x=self.fcmodel(x)
        return x

class PhonemeClassifier(object):
    def __init__(self):
        unique_phonemes = ['CL', 'SF', 'VS', 'WF', 'ST', 'NF', "q"]
        self.labels = unique_phonemes

    def label_to_ids(self, y):
        y_ = [self.labels.index(label) for label in y]
        return y_

    def id_to_label(self, y):
        y_ = [self.labels[i] for i in y]
        return y_
        
    def train(self, X_train, y_train):
        y_train = self.label_to_ids(y_train)
        y_train = np.asarray(y_train)
        ### BEGIN YOUR CODE
        X_train=torch.from_numpy(X_train).float()
        X_train=X_train.reshape((-1,13,3))
        y_train=torch.Tensor(y_train).long()
        print(X_train.shape,y_train.shape)
        dataset=TensorDataset(X_train,y_train)
        dataset=DataLoader(dataset,batch_size=4096,shuffle=True)
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        model=Net()
        model=model.to(device)
        loss_fn=nn.CrossEntropyLoss()
        loss_fn=loss_fn.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
        model.train()
        for epoch in range(20):
            loss_sum=0
            for data in dataset:
                x,y=data
                x=x.to(device)
                y=y.to(device)
                y_pred=model(x)
                loss=loss_fn(y_pred,y)
                loss_sum+=loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Epoch: {}, Loss_sum: {}".format(epoch,loss_sum))
        ### END YOUR CODE
        self.model = model

    def test(self, X_test, y_test):
        """
        @arguments:
        X_test: #frames, #features (39 for mfcc)
        y_test: frame labels
        """
        ### BEGIN YOUR CODE
        # perform prediction and get out_classes array
        # which contain class label id for each frame.
        X_test=torch.from_numpy(X_test).float()
        X_test=X_test.reshape((-1,13,3))
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        X_test=X_test.to(device)
        self.model.eval()
        label_predict=self.model(X_test)
        out_classes=torch.argmax(label_predict,axis=1)
        ### END YOUR CODE
        
        out_classes = self.id_to_label(out_classes) # from id to string
        out_classes = np.asarray(out_classes)
        acc = sum(out_classes == y_test) * 1.0 / len(out_classes)
        print('Frame-based Accuracy using %d testing samples: %f' % (X_test.shape[0], acc))

        return out_classes
