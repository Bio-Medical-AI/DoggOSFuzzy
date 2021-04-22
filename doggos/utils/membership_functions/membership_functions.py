from numpy import exp, minimum, maximum
from numpy.core.function_base import linspace


def gaussian(domain, mean, sigma, max_value=1):
    """Gaussian membership function

    Defines membership function of gaussian distribution shape
    by a vector of values. To be used to determine membership value of crisp number
    to fuzzy set defined by this function.

    Args:
      domain:
        Array-like input which indicate the points from universe for which
        the membership function will be evaluated.
      mean:
        Center of gaussian function, expected value.
      sigma:
        Standard deviation of gaussian function
      max_value:
        Maximum value of membership function, height.

    Returns:
      ndarray
      A vector of membership values corresponding to input. Number of values
      defined implicitly by domain argument. Each value contains calculated membership degree
      to fuzzy set defined by the membership function.


    Example usage:
      >>> x = linspace(0, 1, 201)
      >>> gaussian_set = gaussian(x, 0.4, 0.15, 1)
    """
    return max_value * exp(-(((mean - domain) ** 2) / (2 * sigma ** 2)))


def sigmoid(domain, offset, magnitude):
    """Sigmoid membership function

    Defines membership function of sigmoid shape
    by a vector of values. To be used to determine membership value of crisp number
    to fuzzy set defined by this function.

    Args:
      domain:
        Array-like input which indicate the points from universe for which
        the membership function will be evaluated.
      offset:
        Offset, bias is the center value of the sigmoid, where it has value of 0.5.
        Determines 'lean' of function.
      magnitude:
        Defines width of sigmoidal region around offset. Sign of the value determines which side
        of the function is open.

    Returns:
      ndarray
      A vector of membership values corresponding to input. Number of values
      defined implicitly by domain argument. Each value contains calculated membership degree
      to fuzzy set defined by the membership function.


    Example usage:
      >>> x = linspace(0, 1, 100)
      >>> sigmoid_set = sigmoid(x1, 0.5, -15)
    """
    return 1. / (1. + exp(- magnitude * (domain - offset)))


def triangular(domain, l_end, center, r_end, max_value=1):
    """Triangular membership function

    Defines membership function of triangular shape
    by a vector of values. To be used to determine membership value of crisp number
    to fuzzy set defined by this function.

    Args:
      domain:
        Array-like input which indicate the points from universe for which
        the membership function will be evaluated.
      l_end:
        Left end, vertex of triangle, where value of function is equal to 0.
      center:
        Top vertex of triangle, where value of function is equal to 1.
      r_end:
        Right end, vertex of triangle, where value of function is equal to 0.
      max_value:
        Maximum value of membership function, height.

    Returns:
      ndarray
      A vector of membership values corresponding to input. Number of values
      defined implicitly by domain argument. Each value contains calculated membership degree
      to fuzzy set defined by the membership function.


    Example usage:
      >>> x = linspace(0, 1, 100)
      >>> triangle_set = triangular(x1, 0.2, 0.3, 0.7)
    """
    return minimum(1, maximum(0, ((max_value * (domain - l_end) / (center - l_end)) * (domain <= center) +
                                  ((max_value * ((r_end - domain) / (r_end - center))) * (domain > center)))))


def trapezoidal(domain, l_end, l_center, r_center, r_end, max_value=1):
    """Triangular membership function

    Defines membership function of trapezoidal shape
    by a vector of values. To be used to determine membership value of crisp number
    to fuzzy set defined by this function.

    Args:
      domain:
        Array-like input which indicate the points from universe for which
        the membership function will be evaluated.
      l_end:
        Left end, vertex of trapezoid, where value of function is equal to 0.
      l_center:
        Top left vertex of trapezoid, where value of function is equal to 1.
      r_center:
        Top right vertex of triangle, where value of function is equal to 1.
      r_end:
        Right end, vertex of trapezoid, where value of function is equal to 0.
      max_value:
        Maximum value of membership function, height.

    Returns:
      ndarray
      A vector of membership values corresponding to input. Number of values
      defined implicitly by domain argument. Each value contains calculated membership degree
      to fuzzy set defined by the membership function.


    Example usage:
      >>> x = linspace(0, 1, 100)
      >>> trapezoid_set = trapezoidal(x1, 0.2, 0.3, 0.6, 0.7)
    """
    return minimum(1, maximum(0, ((((max_value * ((domain - l_end) / (l_center - l_end))) * (domain <= l_center)) +
                                   ((max_value * ((r_end - domain) / (r_end - r_center))) * (domain >= r_center))) +
                                  (max_value * ((domain > l_center) * (domain < r_center))))))
