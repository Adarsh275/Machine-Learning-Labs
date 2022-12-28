# Assume df is a pandas dataframe object of the dataset given
import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float

def get_entropy_of_dataset(df):

    entropy = 0
    Target_Values = df[[df.columns[-1]]].values
    _, counts = np.unique(Target_Values, return_counts=True)
    TotalCount = np.sum(counts)

    for Frequency in counts:
        temporary = Frequency/TotalCount
        if temporary != 0:
            entropy -= temporary*(np.log2(temporary)) # calculate entropy
    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float

def get_avg_info_of_attribute(df, attribute):
   
    AttributeVal = df[attribute].values
    UniqueAttributeVal = np.unique(AttributeVal)
    rows = df.shape[0]
    entropy_of_attribute = 0

    for CurrentValue in UniqueAttributeVal:
        df_slice = df[df[attribute] == CurrentValue]
        target = df_slice[[df_slice.columns[-1]]].values
        _, counts = np.unique(target, return_counts=True)
        TotalCount = np.sum(counts)
        entropy = 0

        for freq in counts:
            temporary = freq/TotalCount
            if temporary != 0:
                entropy -= temporary*np.log2(temporary)
        entropy_of_attribute += entropy*(np.sum(counts)/rows)
    avg_info = abs(entropy_of_attribute) 

    return avg_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float

def get_information_gain(df, attribute):
    
    information_gain = 0
    entropy_of_attribute = get_avg_info_of_attribute(df, attribute)
    entropy_of_dataset = get_entropy_of_dataset(df)
    information_gain = entropy_of_dataset - entropy_of_attribute
    
    return information_gain


#input: pandas_dataframe
#output: ({dict},'str')

def get_selected_attribute(df):

    InformationGain = {}
    SelectedColumn = ''
    
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C') '''

    max_information_gain = float("-inf")
    
    for attribute in df.columns[:-1]:

        information_gain_of_attribute = get_information_gain(df, attribute)
        if information_gain_of_attribute > max_information_gain:
            SelectedColumn = attribute
            max_information_gain = information_gain_of_attribute
        InformationGain[attribute] = information_gain_of_attribute

    return (InformationGain, SelectedColumn)