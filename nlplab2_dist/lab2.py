from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import torch
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.linear_model import LogisticRegression

# A list of labels.
OFFSET = '**OFFSET**'

# deliverable 1.1
def bag_of_words(text):
    words=text.split()
    counter=Counter()
    for e in words:
        counter[e]+=1
    return counter
    

# deliverable 1.2
def aggregate_counts(bags_of_words):
    final_cout=Counter()
    for c in bags_of_words:
        final_cout+=c
    return final_cout

# deliverable 1.3
def compute_oov(bow1, bow2):
    
    return set(bow1)-set(bow2)

# deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):

    words=[e[0] for e in training_counts.items() if (e[1]>=min_counts)]
    res_list=[]
    for counter in target_data:
        dic={}
        for e in counter.items():
            if e[0] in words:
                dic[e[0]]=counter[e[0]]
        res_list.append(Counter(dic))
    return res_list,set(words)


def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())


# hint! use this.
def argmax(scores):
    items = list(scores.items())
    items.sort()
    return items[np.argmax([i[1] for i in items])][0]

# This will no longer work for our purposes since python3's max does not guarantee deterministic ordering
# argmax = lambda x : max(x.items(),key=lambda y : y[1])[0]

# deliverable 2.1
def make_feature_vector(base_features,label):

    dic={}
    for e in base_features.keys():
        dic[(label,e)]=base_features[e]
    dic[(label,OFFSET)]=1
    return dic
# deliverable 2.2
def predict(base_features,weights,labels):

    score_dic={}
    for label in labels:
        score=0
        fv=make_feature_vector(base_features,label)
        for e in fv.keys():
            if e in weights.keys():
                score+=weights[e]*fv[e]
        score_dic[label]=score
    max_label=argmax(score_dic)
    return max_label,score_dic
def predict_all(x,weights,labels):
    y_hat=[]
    for base_features in x:
        y_hat.append(predict(base_features,weights,labels)[0])
    return y_hat
theta_hand = defaultdict(float,
                         {('2000s','money'):0.1,
                          ('2000s','name'):0.2,
                          ('1980s','tonight'):0.1,
                          ('2000s','man'):0.1,
                          ('1990s','fly'):0.1,
                          ('pre-1980',OFFSET):0.1
                         })

# deliverable 3.1
def get_corpus_counts(x,y,label):
#     """Compute corpus counts of words for all documents with a given label.

#     :param x: list of counts, one per instance
#     :param y: list of labels, one per instance
#     :param label: desired label for corpus counts
#     :returns: defaultdict of corpus counts
#     :rtype: defaultdict

#     Example:
#     x = [Counter({'aa': 1, 'bb': 2, 'cc': 3}),
#         Counter({'aa': 1, 'dd': 2, 'ee': 3}),
#         Counter({'bb': 1, 'cc': 2, 'dd': 3})]
#     y = [1, 2, 1]
#     label = 1
#     get_corpus_counts(x,y,label) = {'aa': 1, 'bb': 3, 'cc': 5, 'dd': 3}

#     """
#     raise NotImplementedError
    count=Counter()
    for i in range(len(y)):
        if y[i]==label:
            count.update(x[i])
    return defaultdict(int,count)
    
# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
#     '''
#     Compute smoothed log-probability P(word | label) for a given label.

#     :param x: list of counts, one per instance
#     :param y: list of labels, one per instance
#     :param label: desired label
#     :param smoothing: additive smoothing amount
#     :param vocab: list of words in vocabulary
#     :returns: defaultdict of log probabilities per word
#     :rtype: defaultdict

#     '''
#     raise NotImplementedError
    counts=get_corpus_counts(x,y,label)
    deno=sum(counts.values())+smoothing*len(vocab)
    dic={}
    for e in vocab:
        nume=counts[e]+smoothing
        dic[e]=np.log(nume/deno)
    return defaultdict(int,dic)
# deliverable 3.3
def estimate_nb(x,y,smoothing):
#     """estimate a naive bayes model

#     :param x: list of dictionaries of base feature counts
#     :param y: list of labels
#     :param smoothing: smoothing constant
#     :returns: a defaultdict of features and weights. features are tuples (label,base_feature).
#     :rtype: defaultdict

#     Hint: See predict() for the exact return type information.

#     """

#     raise NotImplementedError
    training_counts=aggregate_counts(x)
    vocab=[e[0] for e in training_counts.items() if (e[1]>=10)]
    dic={}
    for label in y:
        counts=get_corpus_counts(x,y,label)
        deno=sum(counts.values())+smoothing*len(vocab)
        for e in vocab:
            nume=counts[e]+smoothing
            dic[(label,e)]=np.log(nume/deno)
        dic[(label,OFFSET)]=np.log(sum([1 for e in y if e==label])/len(y))
    return defaultdict(float,dic)
            
    

# deliverable 3.4
def find_best_smoother(x_tr_pruned,y_tr,x_dv_pruned,y_dv,smoothers):
#     '''
#     find the smoothing value that gives the best accuracy on the dev data

#     :param x_tr: training instances
#     :param y_tr: training labels
#     :param x_dv: dev instances
#     :param y_dv: dev labels
#     :param smoothers: list of smoothing values
#     :returns: 1) best smoothing value, 2) a dictionary of smoothing values and dev set accuracy.
#     :rtype: 1) float, 2) dictionary

#     '''
#     raise NotImplementedError
    dic={}
    dic_neg={}
    for smoothing in smoothers:
        weights=estimate_nb(x_tr_pruned,y_tr,smoothing)
        y_est=predict_all(x_dv_pruned,weights,y_dv)
        dic[smoothing]=acc(y_est,y_dv)
        dic_neg[smoothing]=dic[smoothing]*-1
    best_smoothing= argmax(dic_neg)
    return best_smoothing,dic
    
    
        

def acc(y_hat,y):
    return (y_hat == y).mean()

def write_predictions(y_hat,filename):
    with open(filename,'w') as fout:
        for y_hat_i in y_hat:
            fout.write(y_hat_i + "\n")

def read_predictions(filename):
    with open(filename,'r') as fin:
        return [line.rstrip() for line in fin.readlines()]

## these are just for fun

def f1(y_hat,y,label):
    tp = sum((y_hat==label) & (y==label))
    fp = sum((y_hat==label) & (y!=label))
    fn = sum((y_hat!=label) & (y==label))
    #print tp,fp,fn
    r = tp/float(tp + fn + 1e-10)
    p = tp/float(tp + fp + 1e-10)
    f = 2 * r * p / (r + p + 1e-10)
    return f

def macro_f1(y_hat,y):
    all_labels = set(y)
    y_hat = np.array(y_hat)
    f1s = {label:f1(y_hat,y,label) for label in all_labels}
    return sum(f1s.values())/len(all_labels),f1s


# deliverable 4.1
def make_numpy(bags_of_words, vocab):
#     '''
#     Convert the bags of words into a 2D numpy array

#     :param bags_of_words: list of Counters
#     :param vocab: pruned vocabulary
#     :returns: the bags of words as a 2D numpy array (length of bags_of_words by length of vocab)
#     :rtype: numpy array
#     '''
#     raise NotImplementedError
    vocab=sorted(list(vocab))  # list will change the order of set vocab, sorted so it maintain same order
    res=np.zeros((len(bags_of_words),len(vocab)))
    for i in range(len(bags_of_words)):
        for j in range(len(vocab)):
            if(vocab[j] in bags_of_words[i]):
                res[i,j]=bags_of_words[i][vocab[j]]
    return res
            
# deliverable 4.2
def better_model():
#     # scikit_log_reg = LogisticRegression()   ## Tune parameters for this function.
#     ### BEGIN SOLUTION
#     ### END SOLUTION
#     raise NotImplementedError
    scikit_log_reg = LogisticRegression()  
    scikit_log_reg.tol=0.05
    scikit_log_reg.C=0.1
    return scikit_log_reg
    


######################### helper code
def train_model(loss, model, X_tr_var, Y_tr_var,
                num_its = 200,
                X_dv_var = None,
                Y_dv_var = None,
                status_frequency=10,
                optim_args = {'lr':0.002,'momentum':0},
                param_file = 'best.params'):

    # initialize optimizer
    optimizer = optim.SGD(model.parameters(), **optim_args)

    losses = []
    accuracies = []

    for epoch in range(num_its):
        # set gradient to zero
        optimizer.zero_grad()
        # run model forward to produce loss
        output = loss.forward(model.forward(X_tr_var),Y_tr_var)
        # backpropagate and train
        output.backward()
        optimizer.step()

        #print(output.item())
        losses.append(output.item())

        # write parameters if this is the best epoch yet
        if X_dv_var is not None:
            # run forward on dev data
            _, Y_hat = model.forward(X_dv_var).max(dim=1)
            # compute dev accuracy
            acc = acc(Y_hat.data.numpy(),Y_dv_var.data.numpy())
            # save
            if len(accuracies) == 0 or acc > max(accuracies):
                state = {'state_dict':model.state_dict(),
                         'epoch':len(accuracies)+1,
                         'accuracy':acc}
                torch.save(state,param_file)
            accuracies.append(acc)

        # print status message if desired
        if status_frequency > 0 and epoch % status_frequency == 0:
            print("Epoch "+str(epoch+1)+": Dev Accuracy: "+str(acc))

    # load parameters of best model
    checkpoint = torch.load(param_file)
    model.load_state_dict(checkpoint['state_dict'])

    return model, losses, accuracies



def plot_results(losses, accuracies):
    fig,ax = plt.subplots(1,2,figsize=[12,2])
    ax[0].plot(losses)
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('iteration');
    ax[1].plot(accuracies);
    ax[1].set_ylabel('dev set accuracy')
    ax[1].set_xlabel('iteration');

# deliverable 5.1
def get_top_features_LR(scikit_log_reg, vocab,label_set,label,k):
    most_indicative_features = []
    least_indicative_features = []
    label_dic={}
    vocab=sorted(vocab)
    c=scikit_log_reg.coef_
    for i in range(len(label_set)):
        if label_set[i]==label:
            c_idx=i
    features=[e[1] for e in sorted(zip(c[c_idx],vocab))]
    most_indicative_features=features[-k:]
    least_indicative_features=features[:k]
    most_indicative_features.reverse()
    least_indicative_features.reverse()
    return most_indicative_features,least_indicative_features
    
    
    ### BEGIN SOLUTION
    ### END SOLUTION
    

# deliverable 5.2
def get_top_features_NB(theta_nb, label_set,label,k):
    most_indicative_features = []
    least_indicative_features = []
    label_dic={}
    for lab,fea in theta_nb.keys():
        if(lab==label):
            label_dic[fea]=theta_nb[(lab,fea)]
    del label_dic[OFFSET]
    features=[e[0] for e in sorted(label_dic.items(), key=lambda item:item[1])]
    most_indicative_features=features[-k:]
    least_indicative_features=features[:k]
    most_indicative_features.reverse()
    least_indicative_features.reverse()
    return most_indicative_features,least_indicative_features
            
# deliverable 6
def get_PRF(Y_hat_dv, Y_dv, label_set, label):
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    TN=0
    TP=0
    FN=0
    FP=0
    for i in range(len(label_set)):
        if label_set[i]==label:
            idx=i
    for i in range(len(Y_hat_dv)):
        if Y_hat_dv[i]!=idx and Y_dv[i]!=idx:
            TN+=1
        if Y_hat_dv[i]==idx and Y_dv[i]==idx:
            TP+=1
        if Y_hat_dv[i]==idx and Y_dv[i]!=idx:
            FP+=1
        if Y_hat_dv[i]!=idx and Y_dv[i]==idx:
            FN+=1
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    f1=2*(precision*recall)/(precision+recall)
    return precision,recall,f1
                