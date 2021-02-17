import numpy as np
import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# You may use this function as you like.
error = lambda y, yhat: np.mean(y!=yhat)

class Question1(object):
    # The sequence in this problem is different from the one you saw in the jupyter notebook. This makes it easier to grade. Apologies for any inconvenience.
    def BernoulliNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a BernoulliNB classifier using the given data.
        
        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        import time 
        classifier=BernoulliNB()
        begin=time.time()
        classifier.fit(traindata,trainlabels)
        end=time.time()
        fittingTime=end-begin
        esttrain=classifier.predict(traindata)
        trainingError=error(esttrain,trainlabels)
        begin2=time.time()
        estval=classifier.predict(valdata)
        end2=time.time()
        valPredictingTime=end2-begin2
        validationError=error(estval,vallabels)
        
        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def MultinomialNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a MultinomialNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        import time 
        classifier=MultinomialNB()
        begin=time.time()
        classifier.fit(traindata,trainlabels)
        end=time.time()
        fittingTime=end-begin
        esttrain=classifier.predict(traindata)
        trainingError=error(esttrain,trainlabels)
        begin2=time.time()
        estval=classifier.predict(valdata)
        end2=time.time()
        valPredictingTime=end2-begin2
        validationError=error(estval,vallabels)
        
        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LinearSVC_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LinearSVC classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        import time 
        classifier=LinearSVC()
        begin=time.time()
        classifier.fit(traindata,trainlabels)
        end=time.time()
        fittingTime=end-begin
        esttrain=classifier.predict(traindata)
        trainingError=error(esttrain,trainlabels)
        begin2=time.time()
        estval=classifier.predict(valdata)
        end2=time.time()
        valPredictingTime=end2-begin2
        validationError=error(estval,vallabels)
        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LogisticRegression_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LogisticRegression classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        import time 
        classifier=LogisticRegression()
        begin=time.time()
        classifier.fit(traindata,trainlabels)
        end=time.time()
        fittingTime=end-begin
        esttrain=classifier.predict(traindata)
        trainingError=error(esttrain,trainlabels)
        begin2=time.time()
        estval=classifier.predict(valdata)
        end2=time.time()
        valPredictingTime=end2-begin2
        validationError=error(estval,vallabels)
        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def NN_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a Nearest Neighbor classifier using the given data.

        Make sure to modify the default parameter.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata              (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels            (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        import time 
        classifier=KNeighborsClassifier(n_neighbors=1)
        begin=time.time()
        classifier.fit(traindata,trainlabels)
        end=time.time()
        fittingTime=end-begin
        esttrain=classifier.predict(traindata)
        trainingError=error(esttrain,trainlabels)
        begin2=time.time()
        estval=classifier.predict(valdata)
        end2=time.time()
        valPredictingTime=end2-begin2
        validationError=error(estval,vallabels)
        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def confMatrix(self,truelabels,estimatedlabels):
        """ Write a function that calculates the confusion matrix (cf. Fig. 2.1 in the notes).

        You may wish to read Section 2.1.1 in the notes -- it may be helpful, but is not necessary to complete this problem.

        Parameters:
        1. truelabels           (Nv, ) numpy ndarray. The ground truth labels.
        2. estimatedlabels      (Nv, ) numpy ndarray. The estimated labels from the output of some classifier.

        Outputs:
        1. cm                   (2,2) numpy ndarray. The calculated confusion matrix.
        """
        cm = np.zeros((2,2))
        # Put your code below
        TP=np.logical_and(truelabels==1,estimatedlabels==1 )
        TN=np.logical_and(truelabels==-1 , estimatedlabels==-1 )
        FP=np.logical_and(truelabels==-1, estimatedlabels==1 )
        FN=np.logical_and(truelabels==1, estimatedlabels==-1 )
        cm[0][0]=TP.sum()
        cm[0][1]=FP.sum()
        cm[1][0]=FN.sum()
        cm[1][1]=TN.sum()
        return cm

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Run the classifier you selected in the previous part of the problem on the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. testError            Float. The reported test error. It should be less than 1.
        3. confusionMatrix      (2,2) numpy ndarray. The resulting confusion matrix. This will not be graded.
        """
        # You can freely use the following line
        
        # Put your code below
        classifier=LogisticRegression()
        classifier.fit(traindata,trainlabels)
        esttest=classifier.predict(testdata)
        testError=error(esttest,testlabels)
        confusionMatrix = self.confMatrix(testlabels, esttest)
        # Do not change this sequence!
        return (classifier, testError, confusionMatrix)

class Question2(object):
    def crossValidationkNN(self, traindata, trainlabels, k):
        """ Write a function which implements 5-fold cross-validation to estimate the error of a classifier with cross-validation with the 0,1-loss for k-Nearest Neighbors (kNN).

        For this problem, take your folds to be 0:N/5, N/5:2N/5, ..., 4N/5:N for cross-validation.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. k                    Integer. The cross-validated error estimates will be outputted for 1,...,k.

        Outputs:
        1. err                  (k+1,) numpy ndarray. err[i] is the cross-validated estimate of using i neighbors (the zero-th component of the vector will be meaningless).
        """
        # Put your code below
        from sklearn.metrics import zero_one_loss
        err=np.zeros(k+1)
        size=int(traindata.shape[0])
        idx=np.array([np.arange(0,size)])
        for i in range(1,k+1):
            classifier=KNeighborsClassifier(n_neighbors=i)
            for j in range(5):
                cross_index=np.array([np.arange(int(size/5*j),int(size/5*(j+1)))])
                left_index=np.concatenate((np.array([np.arange(0,int(size/5*j))]),np.array([np.arange(int(size/5*(j+1)),size)])),axis=1)
                val_data=traindata[cross_index,:][0]
                val_labels=trainlabels[cross_index][0]
                tra_data=traindata[left_index,:][0]
                tra_labels=trainlabels[left_index][0]
                classifier.fit(tra_data,tra_labels)
                est=classifier.predict(val_data)
                err[i]+=zero_one_loss(est,val_labels)/5
        return err

    def minimizer_K(self, traindata, trainlabels, k):
        """ Write a function that calls the above function and returns 1) the output from the previous function, 2) the number of neighbors within  1,...,k  that minimizes the cross-validation error, and 3) the correponding minimum error.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. k                    Integer. The cross-validated error estimates will be outputted for 1,...,k.

        Outputs:
        1. err                  (k+1,) numpy ndarray. The output from crossValidationkNN().
        2. k_min                Integer (np.int64 or int). The number of neighbors within  1,...,k  that minimizes the cross-validation error.
        3. err_min              Float. The correponding minimum error.
        """
        err = self.crossValidationkNN(traindata, trainlabels, k)
        # Put your code below
        k_min=np.argmin(err[1:])+1
        err_min=err[k_min]
        # Do not change this sequence!
        return (err, k_min, err_min)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train a kNN model on the whole training data using the number of neighbors you found in the previous part of the question, and apply it to the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best k value that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        from sklearn.metrics import zero_one_loss
        err, k_min, err_min=self.minimizer_K(traindata,trainlabels,30)
        classifier=KNeighborsClassifier(n_neighbors=k_min)
        classifier.fit(traindata,trainlabels)
        est=classifier.predict(testdata)
        testError=zero_one_loss(est,testlabels)
        # Do not change this sequence!
        return (classifier, testError)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class Question3(object):
    def LinearSVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15}.

        You should seaerch by hand.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        err=[]
        c=[2**i for i in range(-5,16,1) ]
        for i in c:
            classifier=LinearSVC(dual=False,C=i)
            score=cross_val_score(classifier, traindata, trainlabels, cv=5)
            err.append(1-score.mean())
        c_idx=np.argmin(err)
        C_min=c[c_idx]
        min_err=err[c_idx]
     
        # Do not change this sequence!
        return (C_min, min_err)

    def SVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15} and \gamma from 2^{-15},...,2^{3}.

        Use GridSearchCV to perform a grid search.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. gamma_min            Float. The hyper-parameter \gamma that minimizes the validation error.
        3. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        parameters = {'C':[2 ** i for i in range(-5, 16)], 'gamma':[2 ** i for i in range(-15, 4)]}
    
        classifier=GridSearchCV( SVC(), param_grid=parameters,cv=10)
        classifier.fit(traindata,trainlabels)
        C_min=classifier.best_params_['C']
        gamma_min=classifier.best_params_['gamma']
        score= classifier.best_score_
        min_err=1-score
        # Do not change this sequence!
        return (C_min, gamma_min, min_err)

    def LogisticRegression_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-14},...,2^{14}.

        You may either use GridSearchCV or search by hand.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        parameters = {'C':[2 ** i for i in range(-5, 16)]}
    
        classifier=GridSearchCV(LogisticRegression(), param_grid=parameters,cv=10)
        classifier.fit(traindata,trainlabels)
        C_min=classifier.best_params_['C']
        score= classifier.best_score_
        min_err=1-score
        # Do not change this sequence!
        return (C_min, min_err)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train the best classifier selected above on the whole training set.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best classifier that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        from sklearn.metrics import zero_one_loss
        classifier = SVC( C=8.0, gamma=0.125)
        classifier.fit(traindata, trainlabels)
        predict_labels = classifier.predict(testdata)
        testError = error(predict_labels, testlabels)

        # Do not change this sequence!
        return (classifier, testError)
