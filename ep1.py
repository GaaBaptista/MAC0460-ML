# Gabriel Baptista 8941300

def normal_equation_prediction(X, y):
    """
    Calculates the prediction using the normal equation method.
    You should add a new row with 1s.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :return: prediction
    :rtype: np.ndarray(shape=(N, 1))
    """
    rows, cols = np.shape(X)
    Xaux = np.ones((rows, cols + 1))
    Xaux[0:,1:] = X
   
    w = np.matrix(np.dot(Xaux.T,Xaux)).I # (X^T * X)^-1
    w = np.dot(w,Xaux.T) # (X^T * X)^-1 * X^T
    w = np.dot(w,y) # (X^T * X)^-1 * X^T * y
   
    prediction = np.dot(Xaux,w) # calcula predição a partir de w

    return prediction