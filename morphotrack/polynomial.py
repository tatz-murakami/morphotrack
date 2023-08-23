import numpy as np


def partial_derivative_multivariate_polynomial(c):
    """
    Get the coeffient and power of the partial derivateves from the powers and coefficients of the original function.
    Arguments
        c (ndarray): Array of coefficients ordered so that the coefficients for terms of degree i,j are contained in c[i,j]. If c has dimension greater than two the remaining indices enumerate multiple sets of coefficients.
    Return
        coefs (ndarray): The coefficiet matrix after partial derivative. It will return the same size of matrix as c.
    """
    # calculate coefficient multiplication factor that comes from the derivative
    grids = np.meshgrid(*[np.linspace(0,c.shape[i]-1,c.shape[i]).astype(int) for i in range(c.ndim)], indexing='ij')
    grids = np.asarray(grids)
    
    coefs = np.zeros(grids.shape)
    for i in range(c.ndim):
        coefs[i,...] = np.roll(grids[i,...] * c, -1, axis=i)
    
    return coefs


def generate_coefficient_matrix(powers, coefficients, squeeze=True):
    """
    Get the coeffient and power of the partial derivateves from the powers and coefficients of the original function.
    Arguments
        powers (ndarray): NxM. each column is the power of a variate at the position of the row. For example, [[1,2],[3,4]] indicates a0*x*y^3 + a1*x^2*y^4.
        coefficients (ndarray): NxL. the coefficients on each term. For the above example, [1,2] indicates 1*x*y^3 + 2*x^2*y^4.
            if the array is 2D, the function will return the matrix with the size of the last dimension is L.
        squeeze (bool): 
    Return
        c (ndarray): The first deminsion is L. The last two dimension is the same as c in numpy.polynomial.polynomial.polyvalNd.
    """
    ndim = coefficients.ndim
    if ndim == 1:
        coefficients = coefficients[np.newaxis,:]
    c = np.zeros([coefficients.shape[0]]+(np.max(powers,axis=0)+1).tolist())
    
    # In case powers have duplicated term, e.g. [[1,1],[2,2]], simple indexing using powers do not work well.
    # instead, use for loop to sum up the duplicated term. e.g. powers: [[1,2],[1,2]], coeff: [1,2] returns [[0,0,0],[0,0,3]]
    
    for j in range(coefficients.shape[0]):
        for i in range(powers.shape[0]):
            c[(j,)+tuple(powers[i,...])] += coefficients[j, i]
            
    if ndim == 1:
        if squeeze:
            c = np.squeeze(c,axis=0)
        
    return c


def calculate_polynomial_curve_normal(polynomial_pipeline, in_coordinate, regression_model='linearregression'):
    """
    """
    out_dim = polynomial_pipeline.named_steps[regression_model].coef_.shape[0]
    in_dim = 2 # only supports 2 dimensional input

    # get the values of partial derivative on points on UV
    dfdu = np.zeros((in_coordinate.shape[0],out_dim),dtype=float)
    dfdv = np.zeros((in_coordinate.shape[0],out_dim),dtype=float)

    for i in range(out_dim):
        coeff = generate_coefficient_matrix(
            polynomial_pipeline.named_steps['polynomialfeatures'].powers_, 
            polynomial_pipeline.named_steps[regression_model].coef_[i,:]
        )
        deriv_coeff = partial_derivative_multivariate_polynomial(coeff)

        dfdu[:,i] = np.polynomial.polynomial.polyval2d(in_coordinate[:,0], in_coordinate[:,1], deriv_coeff[0,...])
        dfdv[:,i] = np.polynomial.polynomial.polyval2d(in_coordinate[:,0], in_coordinate[:,1], deriv_coeff[1,...])
        
    return np.cross(dfdu,dfdv)    