import numpy as np
from torch import nn
from torchvision import models
from sklearn import svm
from sklearn.externals import joblib
import warnings
from datetime import datetime


def reduce(feat, output_dim=200, alpha=2.5):
    # subsample
    if output_dim <= feat.shape[1]:
        feat = feat[:, 0:output_dim]
    else:
        raise ValueError('output_dim is larger than the codes! ')

    # apply power norm
    def pownorm(x):
        return np.power(np.abs(x), alpha)

    pw = pownorm(feat)
    norm = np.linalg.norm(pw, axis=1)

    if not np.any(norm):
        warnings.warn("Power norm not evaluated due to 0 value norm")
        return feat

    feat = np.divide(pw, norm[:, np.newaxis])
    feat = np.nan_to_num(feat)

    return feat


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.alexnet(pretrained=True)
        new_classifier = nn.Sequential(
            *list(self.model.classifier.children())[:-1])
        self.model.classifier = new_classifier

        self.output_dim = 200
        self.alpha = 2.5

    def forward(self, x):
        fc7 = self.model(x)
        feat = reduce(fc7.cpu().numpy(), self.output_dim, self.alpha)

        # returns numpy array
        return feat


class SVM:
    def __init__(self):
        super().__init__()
        self.model = svm.LinearSVC(C=1.0, dual=True, verbose=True)
        # self.model = svm.SVC(C=1.0, probability=True)

    def get_params(self):
        return self.model.get_params()

    def train(self, train_set, train_lbls):
        tic = datetime.now()
        self.model.fit(train_set, train_lbls)
        # print(self.get_params())
        print('Training score: ' + str(self.model.score(train_set, train_lbls)))
        print("Time elapsed", datetime.now()-tic)

    def test(self, test_set):
        conf = self.model.decision_function(test_set)
        return conf

    def save(self, save_name):
        joblib.dump(self, save_name, compress=6)
