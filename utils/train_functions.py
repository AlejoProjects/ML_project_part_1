from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.impute import IterativeImputer
from .data_visualization import plot_values
from .data_visualization import correlation_matrix
from .data_visualization import eda_info
import pandas as pd
import numpy as np



def total_eda_info(df):
    '''
    Perform Exploratory Data Analysis (EDA) on the entire DataFrame, including visualizations and statistical summaries for each column.
    param df: The DataFrame to analyze.'''
    print("--- Statistical Summary of critical_temp ---")
    columns = np.size(df.columns)
    column_names = df.columns.tolist() 
    for i in range(columns):
        column = df.iloc[:, i]
        eda_info(column,column_names[i]) 
def IQR_outliers(df, target_col):
    '''
    This function identifies and removes outliers from a specified target column in a DataFrame using the Interquartile Range (IQR) method.
    param df: The input DataFrame containing the data.
    param target_col: The name of the target column from which to remove outliers.
    
    '''
    # 1. Calculamos el primer (Q1) y tercer cuartil (Q3)
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)

    # 2. Calculamos el Rango Intercuartílico (IQR)
    IQR = Q3 - Q1

    # 3. Definimos los límites inferior y superior
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 4. Creamos un nuevo DataFrame filtrando las anomalías
    df_clean = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]

    # Mostramos los resultados del filtrado
    print(f"Límite inferior aceptado: {lower_bound:.2f}")
    print(f"Límite superior aceptado: {upper_bound:.2f}")
    print(f"Tamaño original del dataset: {len(df)} muestras")
    print(f"Tamaño después de aplicar IQR: {len(df_clean)} muestras")
    print(f"Se removieron {len(df) - len(df_clean)} valores atípicos.")
    
    return df_clean
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
def iterative_imputation(df, target_col, max_iter=10):
    '''
    Perform iterative imputation on the features of a DataFrame while keeping the target column intact.
    param df: The input DataFrame containing the data.
    param target_col: The name of the target column that should not be imputed.
    param max_iter: The maximum number of imputation rounds to perform (default is 10).
    '''
    
    print("Starting Iterative Imputation (this may take a moment)...")
    X = df.drop(target_col, axis=1)
    Y = df[target_col]
    
    imputer = IterativeImputer(random_state=42, max_iter=max_iter)
    X_imputed_array = imputer.fit_transform(X)
    
    # Convert back to DataFrame
    X_imputed = pd.DataFrame(X_imputed_array, columns=X.columns)
    
    print("Remaining nulls after imputation:", X_imputed.isna().sum().sum())
    
    return X_imputed, Y

def scale_data(X_vals,scaler_type='robust'):
    if scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    return scaler.fit_transform(X_vals)
def filter_colls_corr(dvalues,X_train,X_test,Y_train,threshold):
# Volvemos a hacer el split local para asegurar consistencia
    X_train_copy = X_train.copy()
    X_train_copy['cfinal'] = Y_train    

    target_corr = X_train_copy.corr()['cfinal'].drop('cfinal')
    feature_cond = list(target_corr[abs(target_corr) >= threshold].index)
    #Usamos las dimensiones filtradas para crear los nuevos conjuntos de entrenamiento y prueba
    X_train_filtered = X_train[feature_cond]
    X_test_filtered = X_test[feature_cond]
    print(f"\nAfter correlation filtering:")
    print(f"Features remaining: {X_train_filtered.shape[1]}")
    print(f"Training set shape: {X_train_filtered.shape}")
    print(f"Testing set shape:  {X_test_filtered.shape}")
    correlation_matrix(dvalues,feature_cond,threshold=threshold)
    return X_train_filtered,X_test_filtered,feature_cond