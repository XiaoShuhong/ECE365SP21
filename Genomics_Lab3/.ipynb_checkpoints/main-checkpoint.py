import pandas as pd
import statsmodels.api as sm 
import numpy as np
import statsmodels

class Lab3(object):
    
    def create_data(self,snp_lines) :
        '''
        Input - the snp_lines parsed at the beginning of the notebook
        Output - You should return the 53 x 3902 dataframe
        '''
        #start code here
        data=np.zeros((53,3902))
        column_name=[]
        cur_list=[]
        for i in range(len(snp_lines)):
            cur_list=snp_lines[i].split()
            column_name.append(str(cur_list[0]+':'+cur_list[1]))
            for j in range(9,len(cur_list)):
                if (cur_list[j]=='0/0'):
                    data[j-9][i]=0
                elif(cur_list[j]=='0/1'or cur_list[j]=='1/0'):
                    data[j-9][i]=1
                elif(cur_list[j]=='1/1'):
                    data[j-9][i]=2
                else:
                    data[j-9][i]=np.nan
        df=pd.DataFrame(data,columns=column_name)
        return df
            
            
        
        #end code here

    def create_target(self,header_line) :
        '''
        Input - the header_line parsed at the beginning of the notebook
        Output - a list of values(either 0 or 1)
        '''
        #start code here
        data=header_line.split()[9:]
        phenotype=[]
        for i in range(len(data)):
            if data[i][0:4]=='dark':
                phenotype.append(0)
            else:
                phenotype.append(1)
        return phenotype
        #end code here
    
    def logistic_reg_per_snp(self,df) :
        '''
        Input - snp_data dataframe
        Output - list of pvalues and list of betavalues
        '''
        #start code here
        p=[]
        belta=[]
        head=df.columns[0:-1]
        for i in range(len(head)):
            x=sm.add_constant(df[head[i]])
            y=df[df.columns[-1]]
            classifier=sm.Logit(y,x,missing="drop").fit(method='bfgs',disp='False')
            belta.append(round(classifier.params[1],5))
            p.append(round(classifier.pvalues[1],9))
        return p,belta
        #end code here
    
    
    def get_top_snps(self,snp_data,p_values) :
        '''
        Input - snp dataframe with target column and p_values calculated previously
        Output - list of 5 tuples, each with chromosome and position
        '''
        #start code here
        data=snp_data.columns[0:-1]
        maxium=max(p_values)
        p=[]
        output=[]
        while(len(p)<5):
            id=np.argmin(p_values)
            p.append(id)
            p_values[id]=maxium+1
        for idx in p:
            row=data[idx].split(':')
            output.append((row[0],row[1]))
        return output
            
        
        #end code here