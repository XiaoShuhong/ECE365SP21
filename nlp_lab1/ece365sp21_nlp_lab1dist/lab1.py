from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from nltk.corpus import udhr

def get_freqs(corpus, puncts):
    import re
    freqs = {}
    for i in puncts:
        if i in corpus:
            corpus=corpus.replace(i,' ')
    corpus=re.sub('[0-9]', ' ', corpus)
    corpus=corpus.lower()
    data=corpus.split( )
#print(data[0:100])
    for e in data:
        if e not in freqs:
            freqs[e]=1
        else:
            freqs[e]+=1
    ### BEGIN SOLUTION
    ### END SOLUTION
    return freqs

def get_top_10(freqs):
    top_10 = []
    freqs=sorted(freqs.items(), key=lambda item: item[1])
    top_10=freqs[-10:]
    top_10=[i[0] for i in top_10]
    top_10.reverse()
    ### BEGIN SOLUTION
    ### END SOLUTION
    return top_10

def get_bottom_10(freqs):
    bottom_10 = []
    ### BEGIN SOLUTION
    ### END SOLUTION
    freqs=sorted(freqs.items(), key=lambda item: item[1])
    bottom_10=freqs[:10]
    bottom_10=[i[0] for i in bottom_10]
    return bottom_10

def get_percentage_singletons(freqs):
    ### BEGIN SOLUTION
    ### END SOLUTION
    length= len(freqs)
    singletons=(sum(value == 1 for value in freqs.values()))
    return singletons/length*100

def get_freqs_stemming(corpus, puncts):
    ### BEGIN SOLUTION
    ### END SOLUTION

    import re
    freqs={}
    porter = PorterStemmer()
    for i in puncts:
        if i in corpus:
            corpus=corpus.replace(i,' ')
    corpus=re.sub('[0-9]', ' ', corpus)
    corpus=corpus.lower()
    corpus=corpus.split( )
    data=[porter.stem(e) for e in corpus]
    
    for e in data:
        if e not in freqs:
            freqs[e]=1
        else:
            freqs[e]+=1
   
    return freqs

def get_freqs_lemmatized(corpus, puncts):
    ### BEGIN SOLUTION
    ### END SOLUTION
    import re
    freqs={}
    wordnet_lemmatizer = WordNetLemmatizer()
    for i in puncts:
        if i in corpus:
            corpus=corpus.replace(i,' ')
    corpus=re.sub('[0-9]', ' ', corpus)
    corpus=corpus.lower()
    corpus=corpus.split( )
    data=[wordnet_lemmatizer.lemmatize(e,pos="v") for e in corpus]

    for e in data:
        if e not in freqs:
            freqs[e]=1
        else:
            freqs[e]+=1
    ### BEGIN SOLUTION
    ### END SOLUTION
    return freqs
def size_of_raw_corpus(freqs):
    ### BEGIN SOLUTION
    ### END SOLUTION
  
    return len(freqs)

def size_of_stemmed_raw_corpus(freqs_stemming):
    ### BEGIN SOLUTION
    ### END SOLUTION
    
   
   
    return len(freqs_stemming)

def size_of_lemmatized_raw_corpus(freqs_lemmatized):
   
    return len(freqs_lemmatized)

def percentage_of_unseen_vocab(a, b, length_i):
    return len(set(a)-set(b))/length_i

def frac_80_perc(freqs):
    ### BEGIN SOLUTION
    ### END SOLUTION
    word_num=0
    word_type=0
    word_sum=sum(freqs.values())
    freqs=sorted(freqs.items(), key=lambda item: item[1], reverse=True)
    words_len=len(freqs)
    for key in freqs:
        word_num+=key[1]
        word_type+=1
        if((word_num/word_sum) >= 0.8):
            break  
    frac_val=word_type/words_len
    return frac_val

def plot_zipf(freqs):
    
    x=list(range(1,len(freqs)+1))
    y=[]
    
    freqs=sorted(freqs.items(), key=lambda item: item[1], reverse=True)
    for e in freqs:
        y.append(e[1])
    plt.plot(x,y)
    plt.xlabel('rank of words')
    plt.ylabel('frequency of words')
    plt.show()

def get_TTRs(languages):
    TTRs = {}
    num_tokens=[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]
    for lang in languages:
        words = udhr.words(lang)
        words =[e.lower() for e in words]
        count=[]
        for t in num_tokens:
            type=set(words[:t])
            count.append(len(type))
        TTRs[lang]=count   
    return TTRs
        
def plot_TTRs(TTRs):
    ### BEGIN SOLUTION
    ### END SOLUTION
    import numpy as np
    num_tokens=[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300]
    for key in TTRs:
        plt.plot(num_tokens,TTRs[key],label=key) 
    plt.xlabel(' umber of tokens ')
    plt.ylabel('count of types ')
    plt.legend(loc="upper left")
    plt.xticks(np.arange(100,1400,100))
    plt.show()  # put this line at the end to display the figure.
