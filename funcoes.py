# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:33:27 2019

@author: Gladson
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor



def naonumerico(DF):
    
    DF[DF.columns[1:len(DF.columns)]] = DF[DF.columns[1:len(DF.columns)]].apply(pd.to_numeric, errors='coerce')
    DF = DF.dropna()
    DF.reset_index(drop=True,inplace=True)
    
    return DF


def limites_quartiles(DF):
    
    columns = list(DF)   
    #==========================================================================    
    for i in columns: 
        DF = DF[~(DF[i] <= 0)].dropna()
            
    print('Retirando Outliers via BOXPLOT')
    Q1 = DF.quantile(0.25, axis=0, numeric_only=True, interpolation='linear') 
    Q3 = DF.quantile(0.75, axis=0, numeric_only=True, interpolation='linear') 
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5*IQR
    lim_sup = Q3 + 1.5*IQR   
# =============================================================================
#     if 'Temperatura_Escoria' in columns: 
#         print('Modelo  Escória')
#         
#         DF = DF[~((DF['Temperatura_Escoria'] < 800)|(DF['Temperatura_Escoria'] > 1150))].dropna()
# =============================================================================     
    return lim_sup,lim_inf

def preprecessing(DF,minimas,maximas):
    
    columns = list(DF)   
    #==========================================================================    
    for i in columns: 
        if i != columns[0] :
            DF = DF[~(DF[i] <= 0)].dropna()
            
    print('Retirando Outliers via BOXPLOT')
    
   
    for i in columns: 
        if i != columns[0]:
                
            DF = DF[~ ((DF[i] <minimas[i]) | (DF[i] > maximas[i]))].dropna()
        
    DF.reset_index(drop=True,inplace=True)            
    
# =============================================================================
#     if 'Temperatura_Escoria' in columns: 
#         print('Modelo  Escória')
#         
#         DF = DF[~((DF['Temperatura_Escoria'] < 800)|(DF['Temperatura_Escoria'] > 1150))].dropna()
# =============================================================================     
    return DF,columns

def plot_preprocess(V_Av,columns,DF_plot,DF):
    plt.rcParams['figure.figsize'] = (25,10)
    
    plt.figure(0)  
        
    plt.subplot(1, 3, 1)
        
    X = range(len(DF_plot[columns[V_Av]]))

    plt.plot(X,DF_plot[columns[V_Av]], color="blue")
    plt.ylim(min(DF_plot[columns[V_Av]]),max(DF_plot[columns[V_Av]]))
     
    plt.xlabel('Amostragem')
    plt.ylabel(columns[V_Av])
    plt.grid(True)

    
    plt.subplot(1,3, 2)

    plt.boxplot(DF_plot[columns[V_Av]])
        
    
    plt.subplot(1,3,3)  
    
    X = range(len(DF[columns[V_Av]]))
     

    plt.plot(X,DF[columns[V_Av]], color="blue")
    plt.ylim(min(DF_plot[columns[V_Av]]),max(DF_plot[columns[V_Av]]))
    
    plt.xlabel('Amostragem')
    plt.ylabel(columns[V_Av])
    plt.grid(True)
      

    plt.savefig('PreProcessamento.png',dpi = 100)  
        
    return 


def clusters(N_Clusters,DF,N_Outputs):
    df_parcial = DF.iloc[:,0:len(DF.iloc[0,:])-N_Outputs]
    kmeans = KMeans(n_clusters = N_Clusters, random_state = 1)
    kmeans.fit(DF[df_parcial.columns[1:len(df_parcial.columns)]])
    
    DF['K-Classes'] = kmeans.labels_
    
    DF1 = DF.sort_values(by=['K-Classes'])
    DF1.reset_index(drop=True,inplace=True)
    
    
    DF_M1 = DF1.loc[DF1['K-Classes'] == 0]
    DF_M2 = DF1.loc[DF1['K-Classes'] == 1]
    DF_M3 = DF1.loc[DF1['K-Classes'] == 2]
    
    
    DF_M1.reset_index(drop=True,inplace=True)
    DF_M2.reset_index(drop=True,inplace=True)
    DF_M3.reset_index(drop=True,inplace=True)    
    
    return DF_M1, DF_M2,DF_M3,DF,DF1,kmeans


def modelgeneration(N_Outputs,N_Clusters,N_inputs,DF_M1,DF_M2,DF_M3):
    
    Models = []
    Kernels_DT = []
    Y_predicao_teste = []
    Erro_Models = []
    Erro_Means = []
    Desvio_Models = []
    Desvio_Means = []
    Inp_Train_Models = []
    Out_Train_Models = []
    Inp_Val_Models = []
    Out_Val_Models = []
    
    for i in range(N_Clusters):
        
        Models.append([1]*N_Outputs)
        Y_predicao_teste.append([1]*N_Outputs)    
        Erro_Models.append([1]*N_Outputs)
        Erro_Means.append([1]*N_Outputs)
        Kernels_DT.append([1]*N_Outputs)
        Inp_Train_Models.append([1]*N_Outputs)
        Out_Train_Models.append([1]*N_Outputs)
        Inp_Val_Models.append([1]*N_Outputs)
        Out_Val_Models.append([1]*N_Outputs)
        Desvio_Models.append([1]*N_Outputs)
        Desvio_Means.append([1]*N_Outputs)
        
    
    for outp in range(N_Outputs):
        for clus in range(N_Clusters):
            
            Kernels_DT[clus][outp] = AdaBoostRegressor(DecisionTreeRegressor(max_depth=10),n_estimators=500, random_state=None)    
            
            if clus == 0:
                DB = DF_M1
            elif clus == 1:
                DB = DF_M2 
            elif clus == 2:
                DB = DF_M3 
    
       
            train_set, test_set = train_test_split(DB, test_size=0.2)
     
            inp_train = train_set.iloc[:,0:N_inputs]
            out_train =  train_set.iloc[:,(N_inputs+outp)]
     
            inp_teste = test_set.iloc[:,0:N_inputs]
            out_teste = test_set.iloc[:,(N_inputs+outp)]
            
     
            Models[clus][outp]= Kernels_DT[clus][outp].fit(inp_train,out_train)
            Y_predicao_teste[clus][outp] = np.asarray(Models[clus][outp].predict(inp_teste))
            
            Erro_Models[clus][outp] = 100*abs(out_teste.values.tolist()-Y_predicao_teste[clus][outp])/out_teste.values.tolist()
            Erro_Means[clus][outp] = np.mean(Erro_Models[clus][outp])
            

            Desvio_Models[clus][outp] = abs(out_teste.values.tolist()-Y_predicao_teste[clus][outp])
            Desvio_Means[clus][outp] = np.mean(Desvio_Models[clus][outp])       
            
            Inp_Train_Models[clus][outp] = inp_train
            Out_Train_Models[clus][outp] = out_train
            Inp_Val_Models[clus][outp]   = inp_teste
            Out_Val_Models[clus][outp]   = out_teste
            
            print('Modelo para Cluster %s Output %s Concluido.'% ((clus+1),(outp+1)))
        
    return Models,Y_predicao_teste,Erro_Models,Erro_Means,Desvio_Models,Desvio_Means,Inp_Train_Models,Out_Train_Models,Inp_Val_Models,Out_Val_Models


def plot_validation(V_Av1,cluster,N_points,columns,Y_predicao_teste,Out_Val_Models,Out_Train_Models,Erro_Models):
    plt.rcParams['figure.figsize'] = (25,10)
    

    plt.figure(0)       

        
    plt.subplot(1,3, 1)
    
    X = range(len(Y_predicao_teste[cluster-1][V_Av1][0:N_points]))
    
    plt.plot(X,Out_Val_Models[cluster-1][V_Av1][0:N_points], color="red",label="Dados Reais")
    plt.plot(X, Y_predicao_teste[cluster-1][V_Av1][0:N_points], color="blue",label="Dados do Modelo", linewidth=1)
    plt.ylim(min(Out_Val_Models[cluster-1][V_Av1][0:N_points]),max(Out_Val_Models[cluster-1][V_Av1][0:N_points])) 
    
    plt.xlabel('Amostragem')
    plt.ylabel(columns[V_Av1])
    plt.grid(True)
    plt.title("Decision Tree Regression")
    plt.legend() 
    
    plt.subplot(1, 3, 2)

    plt.scatter(Out_Val_Models[cluster-1][V_Av1][0:N_points], Erro_Models[cluster-1][V_Av1][0:N_points], alpha=0.5, 
                               c= Erro_Models[cluster-1][V_Av1][0:N_points])
    
    plt.xlabel('Amostragem')
    plt.ylabel(columns[V_Av1])
    plt.grid(True)
    plt.title("Dispersão do Erro(%)")   
     
    plt.subplot(1, 3, 3)
    
    plt.hist(Erro_Models[cluster-1][V_Av1], bins=20)
    plt.xlabel(columns[V_Av1])
    plt.ylabel('Frequencia de ocorrencia do Erro')
    
    plt.savefig('Validação_Output.png',dpi=100)
        
    
def histogram_validation(V_Av1,cluster,N_points,columns,Y_predicao_teste,Out_Val_Models,Out_Train_Models,Erro_Models):
    
    plt.rcParams['figure.figsize'] = (7,7)

    plt.figure(0)
    fig, ax1 = plt.subplots()
       
    color = 'tab:blue'
    ax1.set_xlabel(columns[V_Av1])
    ax1.set_ylabel('Frequência', color=color)
    
    ax1.hist( Out_Train_Models[cluster-1][V_Av1], bins='auto')
    
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:gray'
    ax2.set_ylabel('Erro Percentual', color=color)  # we already handled the x-label with ax1
    

    ax2.scatter(Out_Val_Models[cluster-1][V_Av1], Erro_Models[cluster-1][V_Av1], color='k',marker = ".")
    
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    

    plt.savefig('Validação_Erro_Histo.png', dpi = 100)
    
#==============================================================================     