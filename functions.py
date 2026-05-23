from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  

def eda_info(column,column_name):
    '''
    Perform Exploratory Data Analysis (EDA) on a given column of a DataFrame.
    '''
    print("\n--- Statistical Summary of " + column_name + " ---")
    print("\nSkewness:",column.skew())
    print("Kurtosis:", column.kurtosis())
    plt.figure(figsize=(10, 6))
    # Histogram and KDE (Kernel Density Estimate)
    sns.histplot(column, bins=50, kde=True, color='purple', edgecolor='black')
    plt.title(column_name, fontsize=16)
    plt.xlabel('values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

def total_eda_info(df):
    # 2. Basic Statistical EDA for 'critical_temp'
    print("--- Statistical Summary of critical_temp ---")
    columns = np.size(df.columns)
    column_names = df.columns.tolist() 
    for i in range(columns):
        column = df.iloc[:, i]
        eda_info(column,column_names[i]) 

def plot_values(x,y,text,color_used='blue'):
    plt.xlabel(text[0])
    plt.ylabel(text[1])
    plt.title(text[2])
    print("columns: ", np.size(x))
    plt.plot(x, y, marker='o', label=text[3], color=color_used)
    plt.legend()

def sort_values(sk,ku):
    skewness_sort = sk.sort_values(ascending=False)
    kurtosis_sort = ku.sort_values(ascending=False)
    return skewness_sort, kurtosis_sort
def classify_kurtosis(ku_sort,epsilon = 0.5):
    '''
    Classify columns based on their kurtosis values into mesokurtic, leptokurtic, and platykurtic categories.
     param ku_sort: A sorted Series of kurtosis values for each column.
     param epsilon: A small threshold value to determine the classification boundaries (default is 0.
    
    '''
   #mesokurtic stands for values closest 0 

    mesokurtic_condition = (-epsilon <= ku_sort ) & (ku_sort <= epsilon)
    mesokurtic = pd.DataFrame(ku_sort[mesokurtic_condition])
    print("Mesokurtic columns:")
    print(f' {mesokurtic.shape[0]} columns')
    #Leptokurtic stands for values higher than 0
    leptokurtic_condition = ku_sort > epsilon
    leptokurtic = pd.DataFrame(ku_sort[leptokurtic_condition])
    print("Leptokurtic columns:")
    print(f' {leptokurtic.shape[0]} columns')
    #Platykurtic stands for values lower than 0
    platykurtic_condition = (ku_sort < -epsilon) 
    platykurtic =pd.DataFrame( ku_sort[platykurtic_condition])
    print("Platykurtic columns:")
    print(f' {platykurtic.shape[0]} columns')
    return mesokurtic, leptokurtic, platykurtic


def classify_skewness(sk_sort, epsilon=0.5):
    '''
    Classify columns based on their skewness values into left-skewed, right-skewed, and approximately symmetric categories.
    params sk_sort: A sorted Series of skewness values for each column.
    params epsilon: A small threshold value to determine the classification boundaries (default is 0.5).
    '''
    
    # Left-skewed (negatively skewed): Skewness < -0.5
    left_skewed_condition = sk_sort < -epsilon
    left_skewed = pd.DataFrame(sk_sort[left_skewed_condition])
    print("Left-skewed columns:")
    print(f' {left_skewed.shape[0]} columns')
    
    # Right-skewed (positively skewed): Skewness > 0.5
    right_skewed_condition = sk_sort > epsilon
    right_skewed = pd.DataFrame(sk_sort[right_skewed_condition])
    print("Right-skewed columns:")
    print(f' {right_skewed.shape[0]} columns')
    
    # Approximately symmetric: Skewness between -0.5 y 0.5 (INCLUSIVO)
    # Se cambian los operadores a >= y <= para cubrir los valores límite
    symmetric_condition = (sk_sort >= -epsilon) & (sk_sort <= epsilon)
    symmetric = pd.DataFrame(sk_sort[symmetric_condition])
    print("Approximately symmetric columns:")
    print(f' {symmetric.shape[0]} columns')
    
    return left_skewed, right_skewed, symmetric
def mean_imputation(df, columns):
    df[columns] = df[columns].fillna(df[columns].mean())
    return df
def median_imputation(df, columns):
    df[columns] = df[columns].fillna(df[columns].median())
    return df