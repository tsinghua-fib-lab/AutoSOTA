# Principal Components Regression

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
# AndySongRMIT

def PCR(x, y):
    pca_model = PCA()
    data_pca = pca_model.fit_transform(x)
    ratio_result = np.cumsum(pca_model.explained_variance_ratio_)

    # Get the index value whose variance ratio exceeds 0.9
    pca_index = np.where(ratio_result > 0.9)
    min_index = pca_index[0][0]
    data_pca_result = data_pca[:, :min_index + 1]

    model = LinearRegression()
    model.fit(data_pca_result, y)
    coefficients = model.coef_
    
    return coefficients
