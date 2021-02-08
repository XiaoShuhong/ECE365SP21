import numpy as np
import scipy.spatial.distance as dist
from scipy import stats
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Question1(object):
    def bayesClassifier(self,data,pi,means,cov):
        C_inv=np.linalg.inv(cov)
        delta=np.log(pi)+(means.dot(C_inv)).dot(data.T).T-0.5*(np.sum(np.multiply(means.dot(C_inv),means),axis=1).T)
        labels=np.argmax(delta,axis=1)
        return labels

    def classifierError(self,truelabels,estimatedlabels):
        error=0
        dif=truelabels-estimatedlabels
        for i in dif:
            if i!=0:
                error +=1
        return error/estimatedlabels.size


class Question2(object):
    def trainLDA(self,trainfeat,trainlabel):
        nlabels = int(trainlabel.max())+1 # Assuming all labels up to nlabels exist.
        pi = np.zeros(nlabels)            # Store your prior in here
        means = np.zeros((nlabels,trainfeat.shape[1]))            # Store the class means in here
        cov = np.zeros((trainfeat.shape[1],trainfeat.shape[1]))   # Store the covariance matrix in here
        for i in range(0, nlabels):
            pi[i] = trainlabel[trainlabel == i].size / trainlabel.size
            subgroup_total = pi[i] * trainlabel.size
            for j in range(0, trainlabel.size):
                if (trainlabel[j] == i):
                    for k in range(0, trainfeat.shape[1]):
                        means[i, k] += trainfeat[j, k]
            for k in range(0, means.shape[1]):
                means[i, k] /= subgroup_total;
            current_mean = means[i]
            np.reshape(current_mean, (trainfeat.shape[1], 1))
            for j in range(0, trainlabel.size):
                if (trainlabel[j] == i):
                    xi = trainfeat[j]
                    temp = xi - current_mean
                    val = np.array([[temp[0] * temp[0], temp[1] * temp[0]],[temp[1] * temp[0], temp[1] * temp[1]]])
                    cov = cov + val
        cov /= trainlabel.size - nlabels
        # Put your code below

        # Don't change the output!
        return (pi,means,cov)

    def estTrainingLabelsAndError(self,trainingdata,traininglabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        (pi,means,cov)=self.trainLDA(trainingdata,traininglabels)
        esttrlabels=q1.bayesClassifier(trainingdata,pi,means,cov)
        trerror=q1.classifierError(traininglabels,esttrlabels)
        # Don't change the output!
        return (esttrlabels, trerror)

    def estValidationLabelsAndError(self,trainingdata,traininglabels,valdata,vallabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        (pi,means,cov)=self.trainLDA(trainingdata,traininglabels)
        estvallabels=q1.bayesClassifier(valdata,pi,means,cov)
        valerror=q1.classifierError(vallabels,estvallabels)
        # Don't change the output!
        return (estvallabels, valerror)


class Question3(object):
    def kNN(self,trainfeat,trainlabel,testfeat, k):
        delta= dist.cdist(trainfeat, testfeat, 'euclidean')
        index=np.argpartition(delta,k,axis=0)[:k]
        first_k=trainlabel[index]
        [labels]=stats.mode(first_k,axis=0)[0]
        
        return labels
     
    def kNN_errors(self,trainingdata, traininglabels, valdata, vallabels):
        q1 = Question1()
        trainingError = np.zeros(4)
        validationError = np.zeros(4)
        k_array = [1,3,4,5]

        for i in range(len(k_array)):
            esttrlabels=self.kNN(trainingdata,traininglabels,trainingdata,k_array[i])
            estvallabels=self.kNN(trainingdata,traininglabels,valdata,k_array[i])
            trainingError[i]=q1.classifierError(traininglabels,esttrlabels)
            validationError[i]=q1.classifierError(vallabels,estvallabels)
            # Please store the two error arrays in increasing order with k
            # This function should call your previous self.kNN() function.
            # Put your code below
            continue

        # Don't change the output!
        return (trainingError, validationError)

class Question4(object):
    def sklearn_kNN(self,traindata,trainlabels,valdata,vallabels):
        import time
        q1 = Question1()
        classifier = neighbors.KNeighborsClassifier(n_neighbors=1,algorithm="ball_tree")
        begin1=time.time()
        classifier.fit(traindata,trainlabels)
        end1=time.time()
        estvallabels=classifier.predict(valdata)
        valerror=q1.classifierError(vallabels,estvallabels)
        fitTime =end1-begin1
        begin2=time.time()
        classifier.predict(valdata)
        end2=time.time()
        predTime =end2-begin2
        
        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

    def sklearn_LDA(self,traindata,trainlabels,valdata,vallabels):
        import time
        q1 = Question1()
        classifier=LinearDiscriminantAnalysis()
        begin1=time.time()
        classifier.fit(traindata,trainlabels)
        end1=time.time()
        fitTime =end1-begin1
        estvallabels=classifier.predict(valdata)
        valerror=q1.classifierError(vallabels,estvallabels)
        begin2=time.time()
        classifier.predict(valdata)
        end2=time.time()
        predTime =end2-begin2
        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

###
