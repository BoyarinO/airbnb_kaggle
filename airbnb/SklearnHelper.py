import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
# Class to extend the Sklearn classifier
class Helper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def predict(self, x):
        return self.clf.predict(x)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
       return self.clf.fit(x, y).feature_importances_



    def get_oof(self, x_train, y_train, x_test,y_test,enable_test=False):
        # Some useful parameters which will come in handy later on
        ntrain = x_train.shape[0]
        ntest = x_test.shape[0]
        SEED = 0  # for reproducibility
        NFOLDS = 5  # set folds for out-of-fold prediction
        kf = KFold(NFOLDS, random_state=SEED)
        num_class = np.unique(y_train).shape[0]

        oof_train = np.zeros((ntrain,num_class))
        oof_test_skf = np.zeros((ntest,num_class*NFOLDS ))

        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            self.fit(x_tr, y_tr)
            # pred =  self.predict(x_te)
            # prob = self.predict_proba(x_te)
            # hst_prob = np.hstack(prob)
            # Xs = np.hsplit(hst_prob, hst_prob.shape[1] / 12)
            oof_train[test_index] = self.predict_proba(x_te)
            oof_test_skf[:,i*num_class:i*num_class+num_class] = self.predict_proba(x_test)

        if enable_test:
            oof_train_t = self.predict(x_test)
            acc_score = metrics.accuracy_score(y_test,oof_train_t)
            print("Accuracy : %.4g" % acc_score)

        oof_test = np.mean(np.hsplit(oof_test_skf, NFOLDS),axis=0)
        return oof_train, oof_test

