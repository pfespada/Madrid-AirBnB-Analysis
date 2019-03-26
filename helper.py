# Investigate the variance accounted for by each principal component.
#function to plot the principal components as well as the cumulative variance
def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components

    INPUT: pca - the result of instantian of PCA in scikit learn

    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    plt.figure(figsize=(20, 15))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    count=0
    for i in range(num_components):
        count+=1
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
        if count==3:
            break


    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')

    # function to fill the NaN values using the mean
 
def impute_missing(df, col):
    """
    HERE YOU SHOULD BRIEFLY DESCRIBE WHAT THE FUNCTION COMPUTES 
    Args: Data frame and column name
        .....
    Returns: The data frame with the missing values converted to the mean()
        ....
    """
    return df[col].fillna(df[col].mean(),inplace=True)

#funtion to get R2 score in train and test data
def quick_val (model):
    '''
    Creates a scree plot associated with the principal components

    INPUT: model to be validated

    OUTPUT:
            R2 result for train and test data
            
    '''
    
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_score = r2_score(y_train, train_predict)
    test_score = r2_score(y_test, test_predict)
    
    

    return print("In the model {}, The rsquared on the training data was {} and the rsquared on the test data was {}.".format(type(model).__name__,train_score, test_score))

