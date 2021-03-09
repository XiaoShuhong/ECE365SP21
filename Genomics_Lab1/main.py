import numpy as np
from collections import OrderedDict

class Lab1(object):
    def parse_reads_illumina(self,reads) :
        '''
        Input - Illumina reads file as a string
        Output - list of DNA reads
        '''
        #start code here
        dna_reads_illumina=[]
        eol=0
        start=0
        end=0
        startlist=[]
        endlist=[]
        for i in range(len(reads)):
            if reads[i] == '\n':
                eol+=1
                if eol % 4 == 1:
                    start=i+1
                    
                if eol % 4 == 2:
                    end=i
                    
                if end>start and reads[start:end] not in dna_reads_illumina:
                    startlist.append(start)
                    endlist.append(end)
        for i in range(len(startlist)):
            if i %3==0:
                dna_reads_illumina.append(reads[startlist[i]:endlist[i]])
        return dna_reads_illumina
            
        #end code here

    def unique_lengths(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - set of counts of reads
        '''
        #start code here
        read_lenset=[]
        for i in range(len(dna_reads)):
            if len(read_lenset)==0:
                read_lenset.append(len(dna_reads[i]))
            else:
                if len(dna_reads[i]) not in read_lenset:
                     read_lenset.append(len(dna_reads[i]))
        read_set={e for e in read_lenset }
        return read_set
        #end code here

    def check_impurity(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - list of reads which have impurities, a set of impure chars 
        '''
        #start code here
        
        change=0
        for e in dna_reads:
            if e.upper()!=e:
                change=1
             
            
        check_box={'A','C','T','G'}
        if change ==1:
            check_box={'a','c','t','g'}
        impure_reads_illumina=[]
        impure_chars_illumina={'\n'}
        for e in dna_reads:
            set_e={e[i] for i in range(len(e))}
            set_extra=set_e-check_box
            num=len(set_extra)
            if num!=0:
                impure_reads_illumina.append(e)
                
            for i in set_extra:
                    
                impure_chars_illumina.add(i)
               
        impure_chars_illumina.remove('\n')
        return impure_reads_illumina,impure_chars_illumina
                
                
            
            
        
        #end code here

    def get_read_counts(self,dna_reads) :
        '''
        Input - list of dna reads
        Output - dictionary with key as read and value as the no. of times it occurs
        '''
        #start code here
        reads_counts_illumina={}
        for e in dna_reads: 
            if e not in reads_counts_illumina.keys():
                reads_counts_illumina[e]=1
            else:
                reads_counts_illumina[e]+=1
        return reads_counts_illumina
        #end code here

    def parse_reads_pac(self,reads_pac) :
        '''
        Input - pac bio reads file as a string
        Output - list of dna reads
        '''
        #start code here
        dna_reads_pac=[]
        head=[]
        data=[]
        rawdata=reads_pac.split('\n')
        for e in rawdata:
            if len(e)!=0:
                data.append(e)
        length=len(data)
        for i in range(length):
            if data[i][0] == '>':
                head.append(i)
                
        string=''
        for i in range(length):
            if i in head and string!='':
                dna_reads_pac.append(string)
                string=''
            elif i not in head:
                string+=data[i]
        dna_reads_pac.append(string)
                
        return dna_reads_pac
                
             
            
            
                
        #end code here