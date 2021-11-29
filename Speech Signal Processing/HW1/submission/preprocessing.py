from sklearn import preprocessing

# Data normalization
def normalize_mean(X):
    """
    Using scikit learn preprocessing to transform feature matrix
    using StandardScaler with mean and standard deviation
    """
    ### BEGIN YOUR CODE
    scaler=preprocessing.StandardScaler()
    X=scaler.fit_transform(X)
    ### END YOUR CODE
    return X, scaler.mean_, np.sqrt(scaler.var_)

def apply_normalize_mean(X, scaler_mean, scaler_std):
    """
    Apply normalizaton to a testing dataset that have been fit using training dataset.
    
    @arguments:
    X: #frames, #features (in case we use mfcc, #features is 39)
    scaler_mean: mean of fitted StandardScaler that you used in normalize_mean function.
    
    @returns:
    X: normalized matrix
    """
    ### BEGIN YOUR CODE
    X=X-scaler_mean
    X=X/scaler_std

    ### END YOUR CODE
    return X
