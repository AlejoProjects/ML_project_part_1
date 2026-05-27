from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
def plot_values(x,y_arr,text,labels_arr,color_used=['blue'],):
    '''
    Plot the relationship between two variables with a line plot.
    param x_arr: A list of independent variables (x-axis).
    param y_arr: A list of dependent variables (y-axis).
    param text: A list containing labels for x, y, and the title.
    param color_used: The color to use for the plot.
    '''
    plt.xlabel(text[0])
    plt.ylabel(text[1])
    plt.title(text[2])
    for i,y in enumerate(y_arr):
      plt.plot(x, y, marker='o', label=labels_arr[i], color=color_used[i])
    plt.legend()
def eda_info(column,column_name):
    '''
    Perform Exploratory Data Analysis (EDA) on a given column of a DataFrame.
    '''
    print("\n--- Statistical Summary of " + column_name + " ---")
    print("\nSkewness:",column.skew())
    print("Kurtosis:", column.kurtosis())
    plt.figure(figsize=(5, 4))
    # Histogram and KDE (Kernel Density Estimate)
    sns.histplot(column, bins=50, kde=True, color='purple', edgecolor='black')
    plt.title(column_name, fontsize=16)
    plt.xlabel('values', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
def linearity_check(u_model, X_test, Y_test):
    '''
    Check for linearity and homoscedasticity using a residuals vs. fitted values plot.
    param u_model: The fitted OLS regression model.
    param X_test: The test set features.
    param Y_test: The test set target variable.
    '''
    # --- Linearity & Homoscedasticity (Visual Check) ---
    print("Checking for Linearity (Visual Check)...")
    fitted = u_model.predict(X_test) # FIXED: Replaced .fittedvalues with .predict()
    residuals = Y_test - fitted
    plt.figure(figsize=(8, 5))
    sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
    plt.title('Residuals vs. Fitted Values')
    plt.xlabel('Fitted Values (Predictions)')
    plt.ylabel('Residuals')
    plt.show()
def model_testing_visualization(model, X_test_scaled, Y_test, color_used='teal', ax=None):
    '''
    Visualize the performance of a regression model by plotting predicted vs. actual values.
    Accepts an 'ax' parameter to allow plotting multiple models in subplots.
    '''
    predictions = model.predict(X_test_scaled)
    
    # If no axis is provided, create a standalone plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        
    ax.scatter(Y_test, predictions, alpha=0.6, color=color_used, edgecolor='k')
    
    # Removed the empty plt.plot() that was doing nothing
    # Ideal identity line (where perfect predictions would fall)
    ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_title(f'Actual vs. Predicted ({model.__class__.__name__})')
    ax.set_xlabel('Actual Values (Kelvin)')
    ax.set_ylabel('Predicted Values (Kelvin)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
def multiple_testing_visualization(models,X_tests,Y_tests,labels,colors):
    for i,mo in enumerate(models):
        predictions = mo.predict(X_tests[i])
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=Y_tests[i], y=predictions, alpha=0.6, color=colors[i], edgecolor='k',label= labels[i])
        sns.scatterplot([Y_tests[i].min(), Y_tests[i].max()], [Y_tests[i].min(), Y_tests[i].max()], 'r--', lw=2, label='Perfect Prediction')
        # Ideal identity line (where perfect predictions would fall)
    plt.plot()
   
    plt.title(f'Actual vs. Predicted Values ({mo.__class__.__name__})')
    plt.xlabel('Actual Values (Kelvin)')
    plt.ylabel('Predicted Values (Kelvin)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
def multiple_model_diagnostics(models, X_test_scaled, Y_test):
    '''
    this function takes a dictionary of fitted regression models and performs a visual diagnostic check for each model by plotting the actual vs. predicted values for the test set.
    param models: A dictionary where keys are model names (strings) and values are fitted regression
    model objects (must have a .predict method).
    param X_test_scaled: The test set features, already scaled if necessary.
    '''
    colors = ['teal', 'blue', 'green']  # Add more colors if you have more models
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    print("--- RENDIMIENTO EN EL SPLIT LOCAL (80-20) ---")
    for i, (name, model) in enumerate(models.items()):
        preds = model.predict(X_test_scaled)
        # Gráfico de Valores Reales vs Predichos
        sns.scatterplot(x=Y_test, y=preds, ax=axes[i], alpha=0.5, color=colors[i])
        axes[i].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2) # Línea ideal
        axes[i].set_title(f'{name})')
        axes[i].set_xlabel('Actual cfinal (Kelvin)')
        if i == 0: axes[i].set_ylabel('Predicted cfinal (Kelvin)')
    plt.suptitle("Diagnóstico de Regresión: Valores Reales vs Predichos (Test Set)", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()
def correlation_matrix(dvalues,features,threshold):
    corr_matrix_full = dvalues.corr()
    plt.figure(figsize=(15, 10))
    keep_cols = list(features) + ['cfinal']
    sns.heatmap(corr_matrix_full.loc[keep_cols, keep_cols], 
            annot=False, 
            cmap='coolwarm', 
            fmt=".2f")
    plt.title(f'Correlation Matrix of Selected Features (Absolute Correlation >= {threshold})')
    plt.tight_layout() 
    plt.show()
def pca_visualization(pca_df,ev_vis):
    plt.figure(figsize=(10, 8))
    # Using a scatterplot colored by the target variable (cfinal)
    scatter = sns.scatterplot(
        x='PC1', y='PC2', 
        hue='cfinal', 
        palette='viridis', 
        data=pca_df, 
        alpha=0.6,
        edgecolor=None
    )
    plt.title('2D PCA of Superconductor Data (Colored by Critical Temp)')
    plt.xlabel(f'Principal Component 1 ({ev_vis[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({ev_vis[1]:.2%} variance)')
    # Move legend out of the way
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='cfinal')
    plt.tight_layout()
    plt.show()
# Instead of 2 components, let's let PCA pick the number of components 
#  needed to explain 95% of the variance in the data.