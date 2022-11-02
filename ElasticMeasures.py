import warnings
import numpy as np
from numba import jit
import math

from sklearn.metrics.pairwise import paired_distances

def dev(X):
    lenx = X.shape[1]
    dx = (2 * X[1:lenx-1] + X[2:lenx] - 3*X[0:lenx-2])/4
    dx = np.insert(dx, 0, dx[0], axis = 1)
    len_dx = dx.shape[1]
    dx = np.insert(dx, len_dx, dx[len_dx-1], axis = 1)
    return dx

# Start of DTW
def dtw(x, y, w=100, constraint=None, fast = True):

    r"""Dynamic Time Warping (DTW) [1]_ utilizes dynamic programming to find 
    the optimal alignment between elements of times series :math:`X = (x_{1}, x_{2}, ..., x_{n})` 
    and :math:`Y = (y_{1}, y_{2}, ..., y_m)` by constructing 
    a distance matrix :math:`M` of shape :math:`(n, m)` with the following forumla:

    .. math::

        M_{i, j} = \begin{cases}
            0, \ i = j = 0 \\
            d_{x_i, y_j} + min \begin{cases}
                M_{i-1, j}\\
                M_{i, j-1} \\
                M_{i-1, j-1}\\
                \end{cases} where \ d_{x_i, y_j}=|x_i - y_j|
        \end{cases}
    
    and return  the DTW distance :math:`M_{n, m}`.

    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100.
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``
    :type fast: bool, optional
    :return: DTW distance
    

    **Example:**
    
    Input:

    .. code-block:: python

        from tsdistance.ElasticMeasures import dtw
        import numpy as np

        X = np.array([3, 2, 4, 5, 5, 2, 4, 7, 9, 8])
        Y = np.array([3, 3, 3, 1, 6, 9, 9])

        dtw_dist = dtw(X, Y, 'Sakoe-Chiba', 3)
        dtw_dist

    Output:

    .. code-block:: bash

      4.123105625617661

    **References**

    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """

    if constraint == "None":
        if fast == True:
            return dtw_n_numba(x, y)
        if fast == False:
            return dtw_n(x, y)
    elif constraint == "Sakoe-Chiba":
        if fast == True:
            return dtw_scb_numba(x, y, w)
        if fast == False:
            return dtw_scb(x, y, w)
    elif constraint == "Itakura":
        if fast == True:
            return dtw_ip_numba(x, y, w)
        if fast == False:
            return dtw_ip(x, y, w)
    else:
        return dtw_n_numba(x, y)

def dtw_n(x, y):

    N = np.zeros((len(x), len(y)))
    N[0][0] = abs(x[0] - y[0]) ** 2

    for i in range(1, len(x)):
        N[i][0] = abs(x[i] - y[0]) ** 2 + N[i - 1][0]

    for i in range(1, len(y)):
        N[0][i] = abs(x[0] - y[i]) ** 2 + N[0][i - 1]

    for i in range(1, len(x)):
        for j in range(1, len(y)):
            if N[i][j] != np.Inf:
                N[i][j] = abs(x[i] - y[j]) ** 2 + min(
                    min(N[i - 1][j], N[i][j - 1]), N[i - 1][j - 1]
                )

    final_dtw = N[len(x) - 1][len(y) - 1]
    return final_dtw ** (1 / 2)


def dtw_ip(x, y, slope):

    cur = np.full(len(y), np.inf)
    prev = np.full(len(y), np.inf)
    xlen = len(x)
    ylen = len(y)

    min_slope = (1 / slope) * (float(ylen) / float(xlen))
    max_slope = slope * (float(ylen) / float(xlen))

    for i in range(len(x)):
        temp = prev
        prev = cur
        cur = temp
        minw = np.ceil(
            max(min_slope * i, ((ylen - 1) - max_slope * (xlen - 1) + max_slope * i))
        )
        maxw = np.floor(
            min(max_slope * i, ((ylen - 1) - min_slope * (xlen - 1) + min_slope * i))
            + 1
        )
        for j in range(int(minw), int(maxw)):

            if i + j == 0:
                cur[j] = abs(x[0] - y[0]) ** 2
            elif i == 0:
                cur[j] = abs(x[0] - y[j]) ** 2 + cur[j - 1]
            elif j == 0:
                cur[j] = abs(x[i] - y[0]) ** 2 + prev[j]
            else:
                cur[j] = abs(x[i] - y[j]) ** 2 + min(prev[j - 1], prev[j], cur[j - 1])
    final_dtw = cur[len(y) - 1]

    return final_dtw ** (1 / 2)


def dtw_scb(x, y, w):

    leny = len(y)
    lenx = len(x)
    cur = np.full(leny, np.inf)
    prev = np.full(leny, np.inf)
    for i in range(lenx):
        minw = max(0, i - w)
        maxw = min(leny, i + w)
        temp = prev
        prev = cur
        cur = temp
        for j in range(int(minw), int(maxw)):

            if i + j == 0:
                cur[j] = abs(x[0] - y[0])
            elif i == 0:
                cur[j] = abs(x[0] - y[j]) + cur[j - 1]
            elif j == 0:
                cur[j] = abs(x[i] - y[0])+ prev[j]
            else:
                cur[j] = abs(x[i] - y[j]) + min(prev[j - 1], prev[j], cur[j - 1])
    
    final_dtw = cur[leny - 1]
    return final_dtw



#ddtw does NOT include "derivative" portion, which is done in data preprocessing step

@jit(nopython=True)
def ddtw(x, y, w):

    N = len(x)
    M = len(y)
    D = np.full((N+1, M+1), np.inf)
    D[0, 0] = 0
    for i in range(1, N+1):
        for j in range(max(1, i-w), min(i+w, M)+1):
            cost = (x[i-1] - y[j-1])**2
            D[i, j] = cost + min(D[i-1,j],D[i-1,j-1],D[i,j-1])

    Dist = math.sqrt(D[N, M])

    return Dist


@jit(nopython=True)
def dtw_n_numba(x, y):

    N = np.zeros((len(x), len(y)))
    N[0][0] = abs(x[0] - y[0]) ** 2

    for i in range(1, len(x)):
        N[i][0] = abs(x[i] - y[0]) ** 2 + N[i - 1][0]

    for i in range(1, len(y)):
        N[0][i] = abs(x[0] - y[i]) ** 2 + N[0][i - 1]

    for i in range(1, len(x)):
        for j in range(1, len(y)):
            if N[i][j] != np.Inf:
                N[i][j] = abs(x[i] - y[j]) ** 2 + min(
                    min(N[i - 1][j], N[i][j - 1]), N[i - 1][j - 1]
                )

    final_dtw = N[len(x) - 1][len(y) - 1]
    return final_dtw ** (1 / 2)


@jit(nopython=True)
def dtw_ip_numba(x, y, slope):

    cur = np.full(len(y), np.inf)
    prev = np.full(len(y), np.inf)
    xlen = len(x)
    ylen = len(y)

    min_slope = (1 / slope) * (float(ylen) / float(xlen))
    max_slope = slope * (float(ylen) / float(xlen))

    for i in range(len(x)):
        temp = prev
        prev = cur
        cur = temp
        minw = np.ceil(
            max(min_slope * i, ((ylen - 1) - max_slope * (xlen - 1) + max_slope * i))
        )
        maxw = np.floor(
            min(max_slope * i, ((ylen - 1) - min_slope * (xlen - 1) + min_slope * i))
            + 1
        )
        for j in range(int(minw), int(maxw)):

            if i + j == 0:
                cur[j] = abs(x[0] - y[0]) ** 2
            elif i == 0:
                cur[j] = abs(x[0] - y[j]) ** 2 + cur[j - 1]
            elif j == 0:
                cur[j] = abs(x[i] - y[0]) ** 2 + prev[j]
            else:
                cur[j] = abs(x[i] - y[j]) ** 2 + min(prev[j - 1], prev[j], cur[j - 1])
    final_dtw = cur[len(y) - 1]

    return final_dtw ** (1 / 2)


@jit(nopython=True)
def dtw_scb_numba(x, y, w):
    N = len(x)
    M = len(y)
    D = np.full((N+1, M+1), np.inf)
    D[0, 0] = 0
    for i in range(1, N+1):
        for j in range(max(1, i-w), min(i+w, M)+1):
            cost = (x[i-1] - y[j-1])**2
            D[i, j] = cost + min(D[i-1,j],D[i-1,j-1],D[i,j-1])

    Dist = math.sqrt(D[N, M])

    return Dist

'''
@jit(nopython=True)
def dtw_scb_numba(x, y, w):

    leny = len(y)
    lenx = len(x)
    cur = np.full(leny, np.inf)
    prev = np.full(leny, np.inf)
    for i in range(lenx):
        minw = max(0, i - w)
        maxw = min(leny-1, i + w)
        temp = prev
        prev = cur
        cur = np.full(leny, np.inf)
        for j in range(minw, maxw+1):

            if i + j == 0:
                cur[j] = (x[0] - y[0]) ** 2
            elif i == 0:
                cur[j] = (x[0] - y[j]) ** 2 + cur[j - 1]
            elif j == 0:
                cur[j] = (x[i] - y[0]) **2 + prev[j]
            else:
                cur[j] = (x[i] - y[j])**2 + min(prev[j - 1], prev[j], cur[j - 1])
    
    final_dtw = math.sqrt(cur[leny - 1])
    return final_dtw
'''

'''
@jit(nopython=True)
def dtw_scb_numba(x, y, w):

    leny = len(y)
    lenx = len(x)
    cur = np.full(leny, np.inf)
    prev = np.full(leny, np.inf)
    for i in range(lenx):
        minw = max(0, i - w)
        maxw = min(leny-1, i + w)
        temp = prev
        prev = cur
        cur = temp
        for j in range(minw, maxw+1):

            if i + j == 0:
                cur[j] = (x[0] - y[0]) ** 2
            elif i == 0:
                cur[j] = (x[0] - y[j]) ** 2 + cur[j - 1]
            elif j == 0:
                cur[j] = (x[i] - y[0]) **2 + prev[j]
            else:
                cur[j] = (x[i] - y[j])**2 + min(prev[j - 1], prev[j], cur[j - 1])
    
    final_dtw = math.sqrt(cur[leny - 1])
    return final_dtw
'''

# End of DTW

#Start of LCSS
def lcss(x,y,epsilon,w = 100, constraint=None, fast=True):

    r"""
    Longest Common Subsequence (LCSS) [1]_ defines similarity by counting the number of "matches" between two time series, 
    where a match is added when the difference between two elements is less than a constant matching threshold, 
    :math:`\epsilon`, which must be tuned to each dataset. 

    This approach allows time series to stretch and match without rearranging the sequence of elements. 
    Note that LCSS allows some elements to be unmatched, whereas Dynamic Time Warping (DTW) pairs all elements, even the outliers. 
    This property enables LCSS to focus only on similar subsequences (matched elements) 
    in computing similarity and thus makes LCSS more robust to extremely noisy data with many disimilar outliers.

    LCSS differs from the general formula as if it determines a match, 
    it will compute :math:`1 + d_{(i-1,j-1)}` without considering the other directions, 
    and if it determines a miss then it will compute :math:`1 + min(d_{(i-1,j)},d_{(i,j-1))}`. 
    The complete formula for LCSS is: 

    .. math::
 
        \begin{aligned}
            D^u(x,y,\epsilon) & =
            \begin{cases}
                1 & \text{if $|x - y| <= \epsilon$} \\
                0 & \text{else}
            \end{cases}\\
            D^h(x,y,\epsilon) = D^v(x,y,\epsilon) & =
            \begin{cases}
                \infty & \text{if $|x - y|<= \epsilon$} \\
                0 & \text{else}
            \end{cases}\\
            D^d(x,y,\epsilon) & =
            \begin{cases}
                1 & \text{if $|x - y| <= \epsilon$} \\
                \infty & \text{else}
            \end{cases}\\
            \pi(d_{n,m}) & = 1 - \frac{d_{n,m}}{min(n,m)}
        \end{aligned}
    

   The function :math:`\pi` is the percent of elements in the smaller time series that are not part of the longest common subsequence. 
   Thus, if the final output is zero, every element in the shorter time series was part of the longest common subsequence. 
   If the value is one, the longest common subsequence was zero and there is no match between two series. 
   Thus, the range of values of the LCSS is :math:`[0,1]`.
   
   :param X: a time series
   :type X: np.array
   :param Y: another time series
   :type Y: np.array
   :param epsilon: the matching threshold
   :type epsilon: float
   :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
   :type constraint: float, optional
   :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100. 
   :type w: float, optional
   :param fast: whether or not to use fast (Numba) implementation,  default to ``True``.
   :type fast: bool, optional
   :return: LCSS distance

    **Example:**

    Input:

    .. code-block:: python

        from tsdistance.ElasticMeasures import lcss
        import numpy as np

        X = np.array([3, 4, 38, 4, 5])
        Y = np.array([0, 3, 4])

        lcss_dist = lcss(X, Y, epsilon = 0.7)
        lcss_dist

    Output:

    .. code-block:: bash

        0.33333333333333337
      
   
   
   
   **References**

   .. [1] Michail  Vlachos,  George  Kollios,  and  Dimitrios  Gunopulos.  
          “DiscoveringSimilar Multidimensional Trajectories”. 
          In:Proceedings of the 18th Inter-national Conference on Data Engineering. IEEE Computer Society, USA. (2002)
    """

    if constraint == "None":
        if fast == True:
            return lcss_n_numba(x, y, epsilon)
        if fast == False:
            return lcss_n(x, y, epsilon);
    elif constraint == "Sakoe-Chiba":
        if fast == True:
            return lcss_scb_numba(x, y, epsilon,w)
        if fast == False:
            return lcss_scb(x, y, epsilon, w);
    elif constraint == "Itakura":
        if fast == True:
            return lcss_ip_numba(x, y, epsilon, w)
        if fast == False:
            return lcss_ip(x, y, epsilon, w);
    else:
        return lcss_n_numba(x, y, epsilon);

@jit(nopython=True)
def dlcss(x,y,epsilon):

    x = dev(x)
    y = dev(y)

    cost = 0;
    arr = np.zeros((len(x),len(y)));
    for i in range(len(x)):

        for j in range(0,len(y)):
            if (i + j == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
            elif (i == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = arr[i][j-1];
            elif j == 0:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = arr[i-1][j];
            else:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                    cost = arr[i-1][j-1] + 1;
                elif (arr[i - 1][j] > arr[i][j - 1]):
                    cost = arr[i-1][j];
                else:
                    cost = arr[i][j-1];
            arr[i][j] = cost;

    result = arr[len(x)-1][len(y)-1];

    return 1 - result/min(len(x),len(y));


@jit(nopython=True)
def lcss_n_numba(x,y,epsilon):
    cost = 0;
    arr = np.zeros((len(x),len(y)));
    for i in range(len(x)):

        for j in range(0,len(y)):
            if (i + j == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
            elif (i == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = arr[i][j-1];
            elif j == 0:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = arr[i-1][j];
            else:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                    cost = arr[i-1][j-1] + 1;
                elif (arr[i - 1][j] > arr[i][j - 1]):
                    cost = arr[i-1][j];
                else:
                    cost = arr[i][j-1];
            arr[i][j] = cost;

    result = arr[len(x)-1][len(y)-1];

    return 1 - result/min(len(x),len(y));

@jit(nopython=True)
def lcss_scb_numba(x, y, epsilon, w):
    lenx = len(x)
    leny = len(y)

    D = np.zeros((lenx, leny))

    for i in range(lenx):
        wmin = max(0, i-w)
        wmax = min(leny-1, i+w)

        for j in range(wmin, wmax+1):
            if i + j == 0:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else:
                    D[i][j] = 0
            elif i ==0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i][j-1]
            elif j ==0: 
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = 1
                else: 
                    D[i][j] =  D[i-1][j]
            else:
                if abs(x[i]-y[j]) <= epsilon:
                    D[i][j] = max(D[i-1][j-1]+1,
                                  D[i-1][j],
                                  D[i][j+1])
                else:
                    D[i][j] = max(D[i-1][j-1],
                                  D[i-1][j],
                                  D[i][j+1])
    
    result = D[lenx-1, leny -1]
    return 1 - result/min(len(x),len(y))
            

@jit(nopython=True)
def lcss_ip_numba(x,y,epsilon,slope):
    cur = np.zeros(len(y));
    prev = np.zeros(len(y));
    xlen = len(x);
    ylen = len(y);

    min_slope = (1/slope) * (float(ylen)/float(xlen));
    max_slope = slope * (float(ylen)/float(xlen));

    cost = 0;
    for i in range(len(x)):
        temp = prev;
        prev = cur;
        cur = temp;
        minw =  np.ceil(max(min_slope * i,
                    ((ylen-1) - max_slope * (xlen - 1)
                        + max_slope * i)))
        maxw = np.floor(min(max_slope * i,
                   ((ylen - 1) - min_slope * (xlen - 1)
                      + min_slope * i)) + 1);

        for j in range(int(minw),int(maxw)):
            if (i + j == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
            elif (i == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = cur[j-1];
            elif j == 0:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = prev[j];
            else:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                    cost = prev[j-1] + 1;
                elif (prev[j] > cur[j - 1]):
                    cost = prev[j];
                else:
                    cost = cur[j-1];
            cur[j] = cost;

    result = cur[len(y)-1];

    return 1 - result/min(len(x),len(y));


def lcss_n(x,y,epsilon):
    cost = 0;
    arr = np.zeros((len(x),len(y)));
    for i in range(len(x)):

        for j in range(0,len(y)):
            if (i + j == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
            elif (i == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = arr[i][j-1];
            elif j == 0:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = arr[i-1][j];
            else:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                    cost = arr[i-1][j-1] + 1;
                elif (arr[i - 1][j] > arr[i][j - 1]):
                    cost = arr[i-1][j];
                else:
                    cost = arr[i][j-1];
            arr[i][j] = cost;

    result = arr[len(x)-1][len(y)-1];

    return 1 - result/min(len(x),len(y));


def lcss_scb(x,y,delta,epsilon):

    cost = 0;
    arr = np.zeros((len(x),len(y)));
    for i in range(len(x)):
        wmin = max(0,i-delta);
        wmax = min(len(y),i+delta);

        for j in range(int(wmin),int(wmax)):
            if (i + j == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
            elif (i == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = arr[i][j-1];
            elif j == 0:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = arr[i-1][j];
            else:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                    cost = arr[i-1][j-1] + 1;
                elif (arr[i - 1][j] > arr[i][j - 1]):
                    cost = arr[i-1][j];
                else:
                    cost = arr[i][j-1];
            arr[i][j] = cost;

    result = arr[len(x)-1][len(y)-1];
    return 1 - result/min(len(x),len(y));


def lcss_ip(x,y,slope,epsilon):
    cur = np.zeros(len(y));
    prev = np.zeros(len(y));
    xlen = len(x);
    ylen = len(y);

    min_slope = (1/slope) * (float(ylen)/float(xlen));
    max_slope = slope * (float(ylen)/float(xlen));

    cost = 0;
    for i in range(len(x)):
        temp = prev;
        prev = cur;
        cur = temp;
        minw =  np.ceil(max(min_slope * i,
                    ((ylen-1) - max_slope * (xlen - 1)
                        + max_slope * i)))
        maxw = np.floor(min(max_slope * i,
                   ((ylen - 1) - min_slope * (xlen - 1)
                      + min_slope * i)) + 1);

        for j in range(int(minw),int(maxw)):
            if (i + j == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
            elif (i == 0):
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = cur[j-1];
            elif j == 0:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                  cost = 1;
                else:
                  cost = prev[j];
            else:
                if (x[i] - epsilon <= y[j] and x[i] + epsilon >= y[j]):
                    cost = prev[j-1] + 1;
                elif (prev[j] > cur[j - 1]):
                    cost = prev[j];
                else:
                    cost = cur[j-1];
            cur[j] = cost;

    result = cur[len(y)-1];

    return 1 - result/min(len(x),len(y));

#start of ERP
def erp(x, y, m, constraint=None, w=5, fast = True):

    r"""
    Edit Distance with Real Penalty (ERP) [1]_ is another edit distance measure that aims to take the advantages of eing a metric (like most $L_{p}$ norm measures) and allowing temporal shifts. 
    It does this by using Lp-norm distance metrics when comparing two elements or comparing each element to a gap variable, m. 
    Being a metric, ERP makes lower bounding possible through the triangle inequality. 
    This is very useful for pruning through clustering and classfication algorithms.

    ERP provides an advantage over other edit distance measures by providing exact differences between values. 
    Additionally, ERP has no :math:`\epsilon` value to tune. 
    Instead, one has to set a gap variable which is often set to 0 to provide intuitive results.

    Lastly, ERP is also very editable; 
    in the formula below, Euclidean distance is used as the internal distance measure but other measures such as absolute difference are also compatible with ERP. 
    This would change :math:`D` and :math:`\pi` but other metrics might have more desirable properties to certain users.

    .. math::

        \begin{aligned}
            D^u & = 0\\
            D^h(x,y,m) & = (x - m)^2\\
            D^v(x,y,m) & = (y - m)^2\\
            D^d(x,y,m) & = (x - y)^2\\
            \pi(d_{n,m}) & = \sqrt{d_{n,m}}
        \end{aligned}

    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param m: the gap variable
    :type m: float
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100. 
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``.
    :type fast: bool, optional
    :return: ERP distance


    **Example:**

    Input:

    .. code-block:: python

            from tsdistance.ElasticMeasures import lcss
            import numpy as np

            X = np.array([3, 4, 38, 4, 5])
            Y = np.array([0, 3, 4])

            erp_dist = erp(X, Y, m = 0)
            erp_dist

    Output:

    .. code-block:: bash

        34.61213659975356

    **References**

    .. [1] Lei  Chen  and  Raymond  Ng.  “On  The  Marriage  of  Lp-norms  and  EditDistance”. In:Proceedings of the 30th VLDB Conference,Toronto, Canada. (2004)
    
    
    """

    if constraint == "None":
        if fast == True:
            return erp_n_numba(x, y, m)
        if fast == False:
            return erp_n(x, y, m)
    elif constraint == "Sakoe-Chiba":
        if fast == True:
            return erp_scb_numba(x, y, m, w)
        if fast == False:
            return erp_scb(x, y, m, w)
    elif constraint == "Itakura":
        if fast == True:
            return erp_ip_numba(x, y, m, w)
        if fast == False:
            return erp_ip(x, y, m, w)
    else:
        return erp_n(x, y, m)

@jit(nopython=True)
def derp(x, y, m):

    x = dev(x)
    y = dev(y)
    
    df = np.zeros((len(x)+1, len(y)+1))

    for i in range(1, len(y)):
        df[0][i] = df[0][i - 1] - pow(y[i] - m, 2)

    for i in range(1, len(x)):
        df[i][0] = df[i - 1][0] - pow(x[i] - m, 2)

    df[1][1] = 0

    for i in range(1, len(x)):
        for j in range(1, len(y)):
            df[i][j] = max(
                df[i - 1][j - 1] - pow(x[i] - y[j], 2),
                df[i][j - 1] - pow(y[j] - m, 2),
                df[i - 1][j] - pow(x[i] - m, 2),
            )

    return math.sqrt(0 - df[len(x) - 1][len(y) - 1])



@jit(nopython=True)
def werp(x, y, m, g):

    xlen = len(x);
    ylen = len(y);
    weight_vector = [1 / (1 + np.exp(-g * (i - xlen / 2))) for i in
                         range(0, xlen)]
    
    df = np.zeros((len(x)+1, len(y)+1))


    for i in range(1, len(y)):
        df[0][i] = df[0][i - 1] - pow(y[i] - m, 2) * weight_vector[min(i, xlen)]

    for i in range(1, len(x)):
        df[i][0] = df[i - 1][0] - pow(x[i] - m, 2) * weight_vector[i]

    df[1][1] = 0

    for i in range(1, len(x)):
        for j in range(1, len(y)):
            df[i][j] = max(
                df[i - 1][j - 1] - weight_vector[abs(i-j)] * pow(x[i] - y[j], 2),
                df[i][j - 1] - weight_vector[abs(i-j)] * pow(y[j] - m, 2),
                df[i - 1][j] - weight_vector[abs(i-j)] * pow(x[i] - m, 2),
            )

    return math.sqrt(0 - df[len(x) - 1][len(y) - 1])


@jit(nopython=True)
def erp_n_numba(x, y, m):
    
    df = np.zeros((len(x)+1, len(y)+1))

    for i in range(1, len(y)):
        df[0][i] = df[0][i - 1] - pow(y[i] - m, 2)

    for i in range(1, len(x)):
        df[i][0] = df[i - 1][0] - pow(x[i] - m, 2)

    df[1][1] = 0

    for i in range(1, len(x)):
        for j in range(1, len(y)):
            df[i][j] = max(
                df[i - 1][j - 1] - pow(x[i] - y[j], 2),
                df[i][j - 1] - pow(y[j] - m, 2),
                df[i - 1][j] - pow(x[i] - m, 2),
            )

    return math.sqrt(0 - df[len(x) - 1][len(y) - 1])

def erp_n(x, y, m):
    
    df = np.zeros((len(x)+1, len(y)+1))

    for i in range(1, len(y)):
        df[0][i] = df[0][i - 1] - pow(y[i] - m, 2)

    for i in range(1, len(x)):
        df[i][0] = df[i - 1][0] - pow(x[i] - m, 2)

    df[1][1] = 0

    for i in range(1, len(x)):
        for j in range(1, len(y)):
            df[i][j] = max(
                df[i - 1][j - 1] - pow(x[i] - y[j], 2),
                df[i][j - 1] - pow(y[j] - m, 2),
                df[i - 1][j] - pow(x[i] - m, 2),
            )

    return math.sqrt(0 - df[len(x) - 1][len(y) - 1])



@jit(nopython=True)
def erp_ip_numba(x, y, m, slope):

    cur = np.zeros(len(y))
    prev = np.zeros(len(y))
    xlen = len(x)
    ylen = len(y)

    min_slope = (1 / slope) * (float(ylen) / float(xlen))
    max_slope = slope * (float(ylen) / float(xlen))

    for i in range(len(x)):
        minw = np.ceil(
            max(min_slope * i, ((ylen - 1) - max_slope * (xlen - 1) + max_slope * i))
        )
        maxw = np.floor(
            min(max_slope * i, ((ylen - 1) - min_slope * (xlen - 1) + min_slope * i))
            + 1
        )

        temp = prev
        prev = cur
        cur = temp
        for j in range(int(minw), int(maxw)):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = cur[j - 1] - pow(y[j] - m, 2)
            elif j == 0:
                cur[j] = prev[j] - pow(x[i] - m, 2)
            else:
                cur[j] = max(
                    prev[j - 1] - pow(x[i] - y[j], 2),
                    cur[j - 1] - pow(y[j] - m, 2),
                    prev[j] - pow(x[i] - m, 2),
                )

    return math.sqrt(0 - cur[len(y) - 1])

def erp_ip(x, y, m, slope):

    cur = np.zeros(len(y))
    prev = np.zeros(len(y))
    xlen = len(x)
    ylen = len(y)

    min_slope = (1 / slope) * (float(ylen) / float(xlen))
    max_slope = slope * (float(ylen) / float(xlen))

    for i in range(len(x)):
        minw = np.ceil(
            max(min_slope * i, ((ylen - 1) - max_slope * (xlen - 1) + max_slope * i))
        )
        maxw = np.floor(
            min(max_slope * i, ((ylen - 1) - min_slope * (xlen - 1) + min_slope * i))
            + 1
        )

        temp = prev
        prev = cur
        cur = temp
        for j in range(int(minw), int(maxw)):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = cur[j - 1] - pow(y[j] - m, 2)
            elif j == 0:
                cur[j] = prev[j] - pow(x[i] - m, 2)
            else:
                cur[j] = max(
                    prev[j - 1] - pow(x[i] - y[j], 2),
                    cur[j - 1] - pow(y[j] - m, 2),
                    prev[j] - pow(x[i] - m, 2),
                )

    return math.sqrt(0 - cur[len(y) - 1])



@jit(nopython=True)
def erp_scb_numba(x, y, m, w):
    lenx = len(x)
    leny = len(y)

    acc_cost_mat = np.full((lenx, leny), np.inf)

    for i in range(lenx):
        minw = max(0, i - w)
        maxw = min(leny-1, i + w)
        
        for j in range(minw, maxw+1):
            if i + j == 0:
                acc_cost_mat[i, j] = 0
            elif i == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i, j-1] + (y[j]-m)**2
            elif j == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i-1, j] + (x[i]-m)**2
            else:
                acc_cost_mat[i, j] = min(acc_cost_mat[i-1, j-1] + (x[i] - y[j])**2,
                                         acc_cost_mat[i, j-1] + (y[j] - m)**2,
                                         acc_cost_mat[i-1, j] + (x[i]-m)**2)
    
    return math.sqrt(acc_cost_mat[lenx-1, leny-1])


'''
@jit(nopython=True)
def erp_scb_numba(x, y, m, w):
    lenx = len(x)
    leny = len(y)

    acc_cost_mat = np.full((lenx, leny), np.inf)

    for i in range(lenx):
        minw = max(0, i - w)
        maxw = min(leny-1, i + w)
        
        for j in range(minw, maxw+1):
            if i + j == 0:
                acc_cost_mat[i, j] = 0
            elif i == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i, j-1] + abs(y[j]-m)
            elif j == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i-1, j] + abs(x[i]-m)
            else:
                acc_cost_mat[i, j] = min(acc_cost_mat[i-1, j-1] + abs(x[i] - y[j]),
                                         acc_cost_mat[i, j-1] + abs(y[j] - m),
                                         acc_cost_mat[i-1, j] + abs(x[i]-m))
    
    return acc_cost_mat[lenx-1, leny-1]

'''
'''
@jit(nopython=True)
def erp_scb_numba(x, y, m, w):

    cur = np.zeros(len(y))
    prev = np.zeros(len(y))

    for i in range(len(x)):
        minw = max(0, i - w)
        maxw = min(len(y)-1, i + w)
        temp = prev
        prev = cur
        cur = temp

        for j in range(minw, maxw+1):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = cur[j - 1] - abs(y[j] - m)
            elif j == 0:
                cur[j] = prev[j] - abs(x[i] - m)
            else:
                cur[j] = max(
                    prev[j - 1] - abs(x[i] - y[j]),
                    cur[j - 1] - abs(y[j] - m),
                    prev[j] - abs(x[i] - m)
                )

    return abs(0 - cur[len(y) - 1])

'''
def erp_scb(x, y, m, w):

    cur = np.zeros(len(y))
    prev = np.zeros(len(y))

    for i in range(len(x)):
        minw = max(0, i - w)
        maxw = min(len(y), i + w)
        temp = prev
        prev = cur
        cur = temp

        for j in range(int(minw), int(maxw)):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = cur[j - 1] - pow(y[j] - m, 2)
            elif j == 0:
                cur[j] = prev[j] - pow(x[i] - m, 2)
            else:
                cur[j] = max(
                    prev[j - 1] - pow(x[i] - y[j], 2),
                    cur[j - 1] - pow(y[j] - m, 2),
                    prev[j] - pow(x[i] - m, 2),
                )

    return math.sqrt(0 - cur[len(y) - 1])

#Start of EDR
def edr(x, y, m=0, constraint=None, w=100, fast = True):

    r"""
    Edit Distance on Real Sequences (EDR) [1]_ is an edit-based elastic measure. 
    Compared to Longest Common Subsequence (LCSS), EDR does not discriminate which direction to pick based on if the current elements were considered a match. 
    Therefore, it is possible for the current elements to match and for the algorithm to take a horizontal or vertical step which is not possible in LCSS. 
    The intuition behind this method is that EDR aims to capture how many edit operations (delete, insert, substitute) it takes to change one time series into another. 
    To determine if one element is the same as another, a matching threshold :math:`\epsilon` is used in a similar way to LCSS, where a match is added when the difference between two comparing elements is less than :math:`\epsilon`.
    The threshold's ability to quantize differences between comparing elements makes EDR useful for very noisy data as outliers in dataset won't disrupt the overall pattern.

    .. math::
        \begin{aligned}
            D(x,y,\epsilon) & =
            \begin{cases}
                0 & \text{if $|x - y| <= \epsilon$}\\
                1 & \text{else}
            \end{cases}\\
            \pi(d_{n,m}) & = d_{n,m}
        \end{aligned}

    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param m: the matching threshold, default to 0
    :type m: float
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100.
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``.
    :type fast: bool, optional
    :return: EDR distance

    **Example:**

    Input:

    .. code-block:: python

        from tsdistance.ElasticMeasures import edr 
        import numpy as np

        X = np.array([3, 4, 74, 4, 5])
        Y = np.array([0, 3, 4])

        edr_dist = erp(X, Y, m = 4)
        edr_dist

    Output:

    .. code-block:: bash

        3.0

    **Reference**

    .. [1]  Lei Chen, M. Tamer Ozsu, and Vincent Oria. “Robust and Fast SimilaritySearch  for  Moving  Object  Trajectories”.  In:ACM SIGMOD, Baltimore,Maryland, USA(2005)
    
    """

    if constraint == "None":
        if fast == True:
            return edr_n_numba(x, y, m)
        if fast == False:
            return edr_n(x, y, m)
    elif constraint == "Sakoe-Chiba":
        if fast == True:
            return edr_scb_numba(x, y, m, w)
        if fast == False:
            return edr_scb(x, y, m, w)
    elif constraint == "Itakura":
        if fast == True:
            return edr_ip_numba(x, y, m, w)
        if fast == False:
            return edr_ip(x, y, m, w)
    else:
        return edr_n_numba(x, y, m)

@jit(nopython=True)
def dedr(x, y, m):

    x = dev(x)
    y = dev(y)

    df = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        df[i] = -i

    for j in range(len(y)):
        df[i] = -j

    for i in range(len(x)):
        for j in range(len(y)):
            if abs(x[i] - y[j]) <= m:
                s1 = 0
            else:
                s1 = -1

            df[i][j] = max(df[i - 1][j - 1] + s1, df[i][j - 1] - 1, df[i - 1][j] - 1)

    return 0 - df[len(x) - 1][len(y) - 1]

@jit(nopython=True)
def wedr(x, y, m, g):

    xlen = len(x);
    ylen = len(y);

    weight_vector = [1 / (1 + np.exp(-g * (i - xlen / 2))) for i in
                         range(0, xlen)]

    df = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        df[i] = -i * weight_vector[i]

    for j in range(len(y)):
        df[i] = -j * weight_vector[min(i, xlen)]

    for i in range(len(x)):
        for j in range(len(y)):
            if abs(x[i] - y[j]) <= m:
                s1 = 0
            else:
                s1 = -1

            df[i][j] = max(df[i - 1][j - 1] + weight_vector[abs(i-j)] * s1, df[i][j - 1] - weight_vector[abs(i-j)] * 1, df[i - 1][j] - weight_vector[abs(i-j)] * 1)

    return 0 - df[len(x) - 1][len(y) - 1]



@jit(nopython=True)
def edr_ip_numba(x, y, m, slope):

    cur = np.zeros(len(y))
    prev = np.zeros(len(y))
    xlen = len(x)
    ylen = len(y)

    min_slope = (1 / slope) * (float(ylen) / float(xlen))
    max_slope = slope * (float(ylen) / float(xlen))

    for i in range(len(x)):
        minw = np.ceil(
            max(min_slope * i, ((ylen - 1) - max_slope * (xlen - 1) + max_slope * i))
        )
        maxw = np.floor(
            min(max_slope * i, ((ylen - 1) - min_slope * (xlen - 1) + min_slope * i))
            + 1
        )
        temp = prev
        prev = cur
        cur = temp

        for j in range(int(minw), int(maxw)):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = -j
            elif j == 0:
                cur[j] = -i
            else:
                if abs(x[0][i] - y[0][j]) <= m:
                    s1 = 0
                else:
                    s1 = -1

                cur[j] = max(prev[j - 1] + s1, prev[j] - 1, cur[j - 1] - 1)

    return 0 - cur[len(y[0]) - 1]

def edr_ip(x, y, m, slope):

    cur = np.zeros(len(y))
    prev = np.zeros(len(y))
    xlen = len(x)
    ylen = len(y)

    min_slope = (1 / slope) * (float(ylen) / float(xlen))
    max_slope = slope * (float(ylen) / float(xlen))

    for i in range(len(x)):
        minw = np.ceil(
            max(min_slope * i, ((ylen - 1) - max_slope * (xlen - 1) + max_slope * i))
        )
        maxw = np.floor(
            min(max_slope * i, ((ylen - 1) - min_slope * (xlen - 1) + min_slope * i))
            + 1
        )
        temp = prev
        prev = cur
        cur = temp

        for j in range(int(minw), int(maxw)):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = -j
            elif j == 0:
                cur[j] = -i
            else:
                if abs(x[0][i] - y[0][j]) <= m:
                    s1 = 0
                else:
                    s1 = -1

                cur[j] = max(prev[j - 1] + s1, prev[j] - 1, cur[j - 1] - 1)

    return 0 - cur[len(y[0]) - 1]



@jit(nopython=True)
def edr_scb_numba(x, y, m, w):

    cur = np.full((1, len(y)), -np.inf)
    prev = np.full((1, len(y)), -np.inf)

    for i in range(len(x)):
        minw = max(0, i - w)
        maxw = min(len(y)-1, i + w)
        prev = cur
        cur = np.full((1, len(y)), -np.inf)

        for j in range(int(minw), int(maxw)+1):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = -j
            elif j == 0:
                cur[j] = -i
            else:
                if abs(x[i] - y[j]) <= m:
                    s1 = 0
                else:
                    s1 = -1

                cur[j] = max(prev[j - 1] + s1, prev[j] - 1, cur[j - 1] - 1)

    return 0 - cur[len(y) - 1]

'''
@jit(nopython=True)
def edr_scb_numba(x, y, m, w):

    cur = np.zeros(len(y))
    prev = np.zeros(len(y))

    for i in range(len(x)):
        minw = max(0, i - w)
        maxw = min(len(y)-1, i + w)
        temp = prev
        prev = cur
        cur = temp

        for j in range(int(minw), int(maxw)+1):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = -j
            elif j == 0:
                cur[j] = -i
            else:
                if abs(x[i] - y[j]) <= m:
                    s1 = 0
                else:
                    s1 = -1

                cur[j] = max(prev[j - 1] + s1, prev[j] - 1, cur[j - 1] - 1)

    return 0 - cur[len(y) - 1]

'''
def edr_scb(x, y, m, w):

    cur = np.zeros(len(y))
    prev = np.zeros(len(y))

    for i in range(len(x)):
        minw = max(0, i - w)
        maxw = min(len(y), i + w)
        temp = prev
        prev = cur
        cur = temp

        for j in range(int(minw), int(maxw)):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = -j
            elif j == 0:
                cur[j] = -i
            else:
                if abs(x[i] - y[j]) <= m:
                    s1 = 0
                else:
                    s1 = -1

                cur[j] = max(prev[j - 1] + s1, prev[j] - 1, cur[j - 1] - 1)

    return 0 - cur[len(y) - 1]


@jit(nopython=True)
def edr_n_numba(x, y, m):

    df = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        df[i] = -i

    for j in range(len(y)):
        df[i] = -j

    for i in range(len(x)):
        for j in range(len(y)):
            if abs(x[i] - y[j]) <= m:
                s1 = 0
            else:
                s1 = -1

            df[i][j] = max(df[i - 1][j - 1] + s1, df[i][j - 1] - 1, df[i - 1][j] - 1)

    return 0 - df[len(x) - 1][len(y) - 1]

def edr_n(x, y, m):

    df = np.zeros((len(x), len(y)))

    for i in range(len(x)):
        df[i][0] = -i

    for j in range(len(y)):
        df[0][i] = -j

    for i in range(len(x)):
        for j in range(len(y)):
            if abs(x[0][i] - y[0][j]) <= m:
                s1 = 0
            else:
                s1 = -1

            df[i][j] = max(df[i - 1][j - 1] + s1, df[i][j - 1] - 1, df[i - 1][j] - 1)

    return 0 - df[len(x[0]) - 1][len(y[0]) - 1]



# Start of TWED

@jit(nopython=True)
def dist(x, y):
    return (x - y) ** 2


def twed(x, timesx, y, timesy, lamb, nu, constraint=None, w=5, fast = True):

    r"""
    Time Warp Edit Distance (TWED) [1]_ is an elastic measure 
    that aims to combine the merits of DTW and edit distance measures like ERP. 
    Unlike ERP, 
    TWED uses time stamps as part of the algorithm which punishes elements that have very different time stamps. 
    TWED controls the extent of this punishment with the parameter :math:`\nu`. 
    TWED replaces the insert, delete, and replace with delete-X, delete-Y, and match.
    The delete operation has a cost of :math:`\lambda`. 
    TWED is a metric and gains all the benefits of a metric as long as the internal distance function is a metric such as absolute value or Euclidean. 
    However, TWED requires two parameters, :math:`\nu` and r:math:`\lambda`, to be set properly which depend on the distance measure. 
    Additionally, the use of time stamps might be difficult as not all data sets include time stamps for the data.

    .. math::

        \begin{aligned}
            D^u(x,\overline{x},t_x, \overline{t_x},y,\overline{y},t_y,\overline{t_y},\nu,\lambda) & = dist(x,y) \\
            D^v(x,\overline{x},t_x, \overline{t_x},y,\overline{y},t_y,\overline{t_y},\nu,\lambda) & = dist(y,\overline{y}) + \nu * (t_y - \overline{t_y}) + \lambda\\
            D^h(x,\overline{x},t_x, \overline{t_x},y,\overline{y},t_y,\overline{t_y},\nu,\lambda) & = dist(x,\overline{x}) + \nu * (t_x - \overline{t_x}) + \lambda\\
            D^d(x,\overline{x},t_x, \overline{t_x},y,\overline{y},t_y,\overline{t_y},\nu,\lambda) & = dist(x,y)+ dist(\overline{x}, \overline{y}) + \nu * (abs(t_y - \overline{t_y}) + abs(t_x - \overline{t_x}))\\
            \pi(d_{n,m}) & = d_{n,m}
        \end{aligned}

    :param X: a time series
    :type X: np.array
    :param timesx: time stamp of time series :math:`X`
    :type timesx: np.array
    :param Y: another time series 
    :type Y: np.array
    :param timesy: time stamp of time series :math:`Y`
    :type timesy: np.array
    :param lamb: cost of delete operation, :math:`\lambda`.
    :type lamb: float
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100. 
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``.
    :type fast: bool, optional
    :return: TWED distance

    **Example:**

    Input:

    .. code-block:: python

        X = np.array([3, 4, 76, 4, 5])
        Y = np.array([0, 3, 4])
        timesx = np.array([i for i in range(len(X))])
        timesy = np.array([i for i in range(len(Y))])

        twed_distance = twed(X, timesx, Y, timesy, lamb = 2.5, nu = 1, w = 5)
        print(twed_distance)

    Output:

    .. code-block:: bash

        4.5

    **References**

    .. [1] Pierre-Fran ̧cois  Marteau.  “Time  Warp  Edit  Distance  with  Stiffness  Ad-justment  for  Time  Series  Matching”.  In:IEEE Transactions on PatternAnalysis and Machine Intelligence31.306 - 318 (2009)
    """

    if constraint == "None":
        if fast == True:
            return twed_n_numba(x, timesx, y, timesy, lamb, nu)
        if fast == False:
            return twed_n(x, timesx, y, timesy, lamb, nu)
    elif constraint == "Sakoe-Chiba":
        if fast == True:
            return twed_scb_numba(x, timesx, y, timesy, lamb, nu, w)
        if fast == False:
            return twed_scb(x, timesx, y, timesy, lamb, nu, w)
    elif constraint == "Itakura":
        if fast == True:
            return twed_ip_numba(x, timesx, y, timesy, lamb, nu, w)
        if fast == False:
            return twed_ip(x, timesx, y, timesy, lamb, nu, w)
    else:
        return twed_n_numba(x, timesx, y, timesy, lamb, nu)

@jit(nopython=True)
def dtwed(x, timesx, y, timesy, lamb, nu):
    
    x = dev(x)
    y = dev(y)

    dp = np.zeros((len(x), len(y)))
    xlen = len(x)
    ylen = len(y)
    cur = np.zeros(ylen)
    prev = np.zeros(ylen)

    for i in range(0, xlen):
        for j in range(0, ylen):
            if i + j == 0:
                dp[i][j] = math.sqrt(dist(x[i], y[j]))
            elif i == 0:
                c1 = (
                    cur[j - 1]
                    + math.sqrt(dist(y[j - 1], y[j]))
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                dp[i][j] = c1
            elif j == 0:
                c1 = (
                    prev[j]
                    + math.sqrt(dist(x[i - 1], x[i]))
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                dp[i][j] = c1
            else:
                c1 = (
                    prev[j]
                    + math.sqrt(dist(x[i - 1], x[i]))
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                c2 = (
                    cur[j - 1]
                    + math.sqrt(dist(y[j - 1], y[j]))
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                c3 = (
                    prev[j - 1]
                    + math.sqrt(dist(x[i], y[j]))
                    + math.sqrt(dist(x[i - 1], y[j - 1]))
                    + nu
                    * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1]))
                )
                dp[i][j] = min(c1, c2, c3)

    return dp[xlen - 1][ylen - 1]

@jit(nopython=True)
def wtwed(x, timesx, y, timesy, lamb, nu, g):

    dp = np.zeros((len(x), len(y)))
    xlen = len(x)
    ylen = len(y)
    cur = np.zeros(ylen)
    prev = np.zeros(ylen)

    weight_vector = [1 / (1 + np.exp(-g * (i - xlen / 2))) for i in
                         range(0, xlen)]
    

    for i in range(0, xlen):
        for j in range(0, ylen):
            if i + j == 0:
                dp[i][j] = math.sqrt(dist(x[i], y[j]))
            elif i == 0:
                c1 = (
                    cur[j - 1] + weight_vector[j] * (
                    + math.sqrt(dist(y[j - 1], y[j]))
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb)
                )
                dp[i][j] = c1
            elif j == 0:
                c1 = (
                    prev[j] + weight_vector[abs(i-j)] * (
                     math.sqrt(dist(x[i - 1], x[i]))
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb)
                )
                dp[i][j] = c1
            else:
                c1 = (
                    prev[j] +  weight_vector[abs(i-j)] * (
                     math.sqrt(dist(x[i - 1], x[i]))
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb)
                )
                c2 = (
                    cur[j - 1] +  weight_vector[abs(i-j)] * (
                     math.sqrt(dist(y[j - 1], y[j]))
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb)
                )
                c3 = (
                    prev[j - 1] + weight_vector[abs(i-j)] * (
                    + math.sqrt(dist(x[i], y[j]))
                    + math.sqrt(dist(x[i - 1], y[j - 1]))
                    + nu
                    * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1])))
                )
                dp[i][j] = min(c1, c2, c3)

    return dp[xlen - 1][ylen - 1]

@jit(nopython=True)
def twed_n_numba(x, timesx, y, timesy, lamb, nu):

    dp = np.zeros((len(x), len(y)))
    xlen = len(x)
    ylen = len(y)
    cur = np.zeros(ylen)
    prev = np.zeros(ylen)

    for i in range(0, xlen):
        for j in range(0, ylen):
            if i + j == 0:
                dp[i][j] = math.sqrt(dist(x[i], y[j]))
            elif i == 0:
                c1 = (
                    cur[j - 1]
                    + math.sqrt(dist(y[j - 1], y[j]))
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                dp[i][j] = c1
            elif j == 0:
                c1 = (
                    prev[j]
                    + math.sqrt(dist(x[i - 1], x[i]))
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                dp[i][j] = c1
            else:
                c1 = (
                    prev[j]
                    + math.sqrt(dist(x[i - 1], x[i]))
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                c2 = (
                    cur[j - 1]
                    + math.sqrt(dist(y[j - 1], y[j]))
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                c3 = (
                    prev[j - 1]
                    + math.sqrt(dist(x[i], y[j]))
                    + math.sqrt(dist(x[i - 1], y[j - 1]))
                    + nu
                    * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1]))
                )
                dp[i][j] = min(c1, c2, c3)

    return dp[xlen - 1][ylen - 1]

def twed_n(x, timesx, y, timesy, lamb, nu):
    

    dp = np.zeros((len(x), len(y)))
    xlen = len(x)
    ylen = len(y)

    cur = np.zeros(ylen)
    prev = np.zeros(ylen)

    for i in range(0, xlen):
        for j in range(0, ylen):
            if i + j == 0:
                dp[i][j] = math.sqrt(dist(x[i], y[j]))
            elif i == 0:
                c1 = (
                    cur[j - 1]
                    + math.sqrt(dist(y[j - 1], y[j]))
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                dp[i][j] = c1
            elif j == 0:
                c1 = (
                    prev[j]
                    + math.sqrt(dist(x[i - 1], x[i]))
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                dp[i][j] = c1
            else:
                c1 = (
                    prev[j]
                    + math.sqrt(dist(x[i - 1], x[i]))
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                c2 = (
                    cur[j - 1]
                    + math.sqrt(dist(y[j - 1], y[j]))
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                c3 = (
                    prev[j - 1]
                    + math.sqrt(dist(x[i], y[j]))
                    + math.sqrt(dist(x[i - 1], y[j - 1]))
                    + nu
                    * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1]))
                )
                dp[i][j] = min(c1, c2, c3)

    return dp[xlen - 1][ylen - 1]
    

@jit(nopython=True)
def twed_ip_numba(x, timesx, y, timesy, lamb, nu, slope=5):

    cur = np.zeros(len(y))
    prev = np.zeros(len(y))
    xlen = len(x)
    ylen = len(y)

    min_slope = (1 / slope) * (float(ylen) / float(xlen))
    max_slope = slope * (float(ylen) / float(xlen))

    for i in range(len(x)):
        temp = prev
        prev = cur
        cur = temp
        minw = np.ceil(
            max(min_slope * i, ((ylen - 1) - max_slope * (xlen - 1) + max_slope * i))
        )
        maxw = np.floor(
            min(max_slope * i, ((ylen - 1) - min_slope * (xlen - 1) + min_slope * i))
            + 1
        )
        for j in range(int(minw), int(maxw)):
            if i + j == 0:
                cur[j] = math.sqrt(dist(x[i], y[j]))
            elif i == 0:
                c1 = (
                    cur[j - 1]
                    + math.sqrt(dist(y[j - 1], y[j]))
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                cur[j] = c1
            elif j == 0:
                c1 = (
                    prev[j]
                    + math.sqrt(dist(x[i - 1], x[i]))
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                cur[j] = c1
            else:
                c1 = (
                    prev[j]
                    + math.sqrt(dist(x[i - 1], x[i]))
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                c2 = (
                    cur[j - 1]
                    + math.sqrt(dist(y[j - 1], y[j]))
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                c3 = (
                    prev[j - 1]
                    + math.sqrt(dist(x[i], y[j]))
                    + math.sqrt(dist(x[i - 1], y[j - 1]))
                    + nu
                    * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1]))
                )
                cur[j] = min(c1, c2, c3)

    return cur[ylen - 1]

def twed_ip(x, timesx, y, timesy, lamb, nu, slope=5):

    cur = np.zeros(len(y))
    prev = np.zeros(len(y))
    xlen = len(x)
    ylen = len(y)

    min_slope = (1 / slope) * (float(ylen) / float(xlen))
    max_slope = slope * (float(ylen) / float(xlen))

    for i in range(len(x)):
        temp = prev
        prev = cur
        cur = temp
        minw = np.ceil(
            max(min_slope * i, ((ylen - 1) - max_slope * (xlen - 1) + max_slope * i))
        )
        maxw = np.floor(
            min(max_slope * i, ((ylen - 1) - min_slope * (xlen - 1) + min_slope * i))
            + 1
        )
        for j in range(int(minw), int(maxw)):
            if i + j == 0:
                cur[j] = math.sqrt(dist(x[i], y[j]))
            elif i == 0:
                c1 = (
                    cur[j - 1]
                    + math.sqrt(dist(y[j - 1], y[j]))
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                cur[j] = c1
            elif j == 0:
                c1 = (
                    prev[j]
                    + math.sqrt(dist(x[i - 1], x[i]))
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                cur[j] = c1
            else:
                c1 = (
                    prev[j]
                    + math.sqrt(dist(x[i - 1], x[i]))
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                c2 = (
                    cur[j - 1]
                    + math.sqrt(dist(y[j - 1], y[j]))
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                c3 = (
                    prev[j - 1]
                    + math.sqrt(dist(x[i], y[j]))
                    + math.sqrt(dist(x[i - 1], y[j - 1]))
                    + nu
                    * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1]))
                )
                cur[j] = min(c1, c2, c3)

    return cur[ylen - 1]
    

@jit(nopython=True)
def twed_scb_numba(x,timesx, y, timesy, lamb, nu, w=10000):

    xlen = len(x)
    ylen = len(y)
    cur = np.full(ylen, np.inf)
    prev = np.full(ylen, np.inf)


    for i in range(0, xlen):
        prev = cur
        cur = np.full(ylen, np.inf)
        minw = max(0, i - w)
        maxw = min(ylen-1, i + w)
        for j in range(minw, maxw+1):
            if i + j == 0:
                cur[j] = (x[i] - y[j]) **2
            elif i == 0:
                c1 = (
                    cur[j - 1]
                    + (y[j - 1] - y[j]) **2
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                cur[j] = c1
            elif j == 0:
                c1 = (
                    prev[j]
                    + (x[i - 1] - x[i]) **2
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                cur[j] = c1
            else:
                c1 = (
                    prev[j]
                    +(x[i - 1] - x[i]) **2
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                c2 = (
                    cur[j - 1]
                    + (y[j - 1] - y[j])**2
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                c3 = (
                    prev[j - 1]
                    + (x[i] - y[j]) ** 2
                    + (x[i - 1]- y[j - 1]) ** 2
                    + nu
                    * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1]))
                )
                cur[j] = min(c1, c2, c3)

    return cur[ylen - 1]

'''
@jit(nopython=True)
def twed_scb_numba(x,timesx, y, timesy, lamb, nu, w=10000):

    xlen = len(x)
    ylen = len(y)
    cur = np.full(ylen, np.inf)
    prev = np.full(ylen, np.inf)


    for i in range(0, xlen):
        prev = cur
        cur = np.full(ylen, np.inf)
        minw = max(0, i - w)
        maxw = min(ylen-1, i + w)
        for j in range(minw, maxw+1):
            if i + j == 0:
                cur[j] = abs(x[i] - y[j])
            elif i == 0:
                c1 = (
                    cur[j - 1]
                    + abs(y[j - 1] - y[j])
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                cur[j] = c1
            elif j == 0:
                c1 = (
                    prev[j]
                    + abs(x[i - 1] - x[i])
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                cur[j] = c1
            else:
                c1 = (
                    prev[j]
                    +abs(x[i - 1] - x[i])
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                c2 = (
                    cur[j - 1]
                    +abs(y[j - 1] - y[j])
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                c3 = (
                    prev[j - 1]
                    + abs(x[i] - y[j])
                    + abs(x[i - 1]- y[j - 1])
                    + nu
                    * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1]))
                )
                cur[j] = min(c1, c2, c3)

    return cur[ylen - 1]
'''

def twed_scb(x,timesx, y, timesy, lamb, nu, w=10000):

    xlen = len(x)
    ylen = len(y)
    cur = np.full(ylen, np.inf)
    prev = np.full(ylen, np.inf)


    for i in range(0, xlen):
        prev = cur
        cur = np.full(ylen, np.inf)
        minw = max(0, i - w)
        maxw = min(ylen-1, i + w)
        for j in range(minw, maxw+1):
            if i + j == 0:
                cur[j] = abs(x[i] - y[j])
            elif i == 0:
                c1 = (
                    cur[j - 1]
                    + abs(y[j - 1] - y[j])
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                cur[j] = c1
            elif j == 0:
                c1 = (
                    prev[j]
                    + abs(x[i - 1] - x[i])
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                cur[j] = c1
            else:
                c1 = (
                    prev[j]
                    +abs(x[i - 1] - x[i])
                    + nu * (timesx[i] - timesx[i - 1])
                    + lamb
                )
                c2 = (
                    cur[j - 1]
                    +abs(y[j - 1] - y[j])
                    + nu * (timesy[j] - timesy[j - 1])
                    + lamb
                )
                c3 = (
                    prev[j - 1]
                    + abs(x[i] - y[j])
                    + abs(x[i - 1]- y[j - 1])
                    + nu
                    * (abs(timesx[i] - timesy[j]) + abs(timesx[i - 1] - timesy[j - 1]))
                )
                cur[j] = min(c1, c2, c3)

    return cur[ylen - 1]


#Start of Move-Split-Merge

def msm(x,y,c,constraint=None,w=5, fast = True):

    r"""
    Move-Split-Merge (MSM) [1]_ is an edit distance measure that deconstructs the popular editing operations (insert, delete, and substitute);
    instead it proposes sub-operations that have can be used in conjunctions to replicate the original operations. 
    Move functions identical to a substitute, changing the value of an element. 
    Merge combines two equal elements in a series into one. 
    Split takes an element creates a duplicate adjacent to it. 
    Thus, insert can be seen as a split-move operation and delete can be seen as a merge-move operation. 

    Similar to ERP, 
    MSM has the advantage of being a metric, 
    which allows MSM to be combined with other generic indexing, 
    clustering, and visualization methods designed to in any metric space. 
    However unlike ERP, MSM is invariant based on the choice of the origin. 
    This means that the distance calculated is unaffected by translations 
    (adding the same constant to both time series).

    Each operation has an associated cost:

    .. math::

        \begin{equation*}
            Cost(move) = |x - \overline{x}|
        \end{equation*}
    
    .. math::

        \begin{equation*}
            Cost(split) = Cost(merge) = c
        \end{equation*}
    
    where :math:`c` is a set constant, and :math:`\overline{x}` is the new value.

    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param c: the cost for one *move* or *split** operation 
    :type c: float
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100. 
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``.
    :type fast: bool, optional
    :return: MSM distance

    **Example:**

    Input:

    .. code-block:: python

        X = np.array([3, 4, 76, 4, 5])
        Y = np.array([0, 3, 4])

        msm_distance = msm(X, Y, c = 1, w = 5)
        print(msm_distance)

    Output:

    .. code-block:: bash

        79.0

    **References:**

    .. [1] Alexandra  Stefan,  Vassilis  Athitsos,  and  Gautam  Das.  “The  Move-Split-Merge Metric for Time Series”. In:IEEE Transactions on Knowledge andData Engineering25.1425 – 1438 (2013).
    """
    
    if constraint == "None":
        if fast == True:
            return msm_n_numba(x,y,c)
        if fast == False:
            return msm_n(x,y,c)
    elif constraint == "Sakoe-Chiba":
        if fast == True:
            return msm_scb_numba(x,y,c,w)
        if fast == False:
            return msm_scb(x,y,c,w)
    elif constraint == "Itakura":
        if fast == True:
            return msm_ip_numba(x,y,c,w)
        if fast == False:
            return msm_ip(x,y,c,w)
    else:
        return msm_n_numba(x,y,c);

@jit(nopython=True)
def msm_dist(new, x, y, c):
    if ((x <= new) and (new <= y)) or ((y <= new) and (new <= x)):
        dist = c;
    else:
        dist = c + min(abs(new - x), abs(new - y))

    return dist;

@jit(nopython=True)
def dmsm(x,y,c):

    x = dev(x)
    y = dev(y)

    cost = np.zeros((len(x),len(y)));

    xlen = len(x);
    ylen = len(y);

    cost[0][0] = abs(x[0] - y[0]);

    for i in range(1,len(x)):
        cost[i][0] = cost[i-1][0] + msm_dist(x[i],x[i-1],y[0],c);

    for i in range(1,len(y)):
        cost[0][i] = cost[0][i-1] + msm_dist(y[i], x[0],y[i-1],c);

    for i in range(1,xlen):
        for j in range(1,ylen):
            cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]),
                            cost[i-1][j] + msm_dist(x[i], x[i -1],y[j],c),
                            cost[i][j-1] + msm_dist(y[j], x[i], y[j-1],c));

    return cost[xlen-1][ylen-1];

@jit(nopython=True)
def wmsm(x,y,c, g):

    cost = np.zeros((len(x),len(y)));

    xlen = len(x);
    ylen = len(y);

    weight_vector = [1 / (1 + np.exp(-g * (i - xlen / 2))) for i in
                         range(0, xlen)]

    cost[0][0] = abs(x[0] - y[0]);

    for i in range(1,len(x)):
        cost[i][0] = cost[i-1][0] + msm_dist(x[i],x[i-1],y[0],c) * weight_vector[i];

    for i in range(1,len(y)):
        cost[0][i] = cost[0][i-1] + msm_dist(y[i], x[0],y[i-1],c) * weight_vector[min(i, xlen)];

    for i in range(1,xlen):
        for j in range(1,ylen):
            cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]) * weight_vector[min(abs(i-j), xlen)],
                            cost[i-1][j] + msm_dist(x[i], x[i -1],y[j],c) * weight_vector[min(abs(i-j), xlen)],
                            cost[i][j-1] + msm_dist(y[j], x[i], y[j-1],c)) * weight_vector[min(abs(i-j), xlen)];

    return cost[xlen-1][ylen-1];


@jit(nopython=True)
def msm_n_numba(x,y,c):

    cost = np.zeros((len(x),len(y)));

    xlen = len(x);
    ylen = len(y);

    cost[0][0] = abs(x[0] - y[0]);

    for i in range(1,len(x)):
        cost[i][0] = cost[i-1][0] + msm_dist(x[i],x[i-1],y[0],c);

    for i in range(1,len(y)):
        cost[0][i] = cost[0][i-1] + msm_dist(y[i], x[0],y[i-1],c);

    for i in range(1,xlen):
        for j in range(1,ylen):
            cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]),
                            cost[i-1][j] + msm_dist(x[i], x[i -1],y[j],c),
                            cost[i][j-1] + msm_dist(y[j], x[i], y[j-1],c));

    return cost[xlen-1][ylen-1];

def msm_n(x,y,c):

    cost = np.zeros((len(x),len(y)));

    xlen = len(x);
    ylen = len(y);

    cost[0][0] = abs(x[0] - y[0]);

    for i in range(1,len(x)):
        cost[i][0] = cost[i-1][0] + msm_dist(x[i],x[i-1],y[0],c);

    for i in range(1,len(y)):
        cost[0][i] = cost[0][i-1] + msm_dist(y[i], x[0],y[i-1],c);

    for i in range(1,xlen):
        for j in range(1,ylen):
            cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]),
                            cost[i-1][j] + msm_dist(x[i], x[i -1],y[j],c),
                            cost[i][j-1] + msm_dist(y[j], x[i], y[j-1],c));

    return cost[xlen-1][ylen-1];

@jit(nopython=True)
def msm_ip_numba(x,y,c,slope):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));
    xlen = len(x);
    ylen = len(y);

    min_slope = (1/slope) * (float(ylen)/float(xlen));
    max_slope = slope * (float(ylen)/float(xlen));


    for i in range(len(x)):
        temp = prev;
        prev = cur;
        cur = temp;
        minw =  np.ceil(max(min_slope * i,
                    ((ylen-1) - max_slope * (xlen - 1)
                        + max_slope * i)))
        maxw = np.floor(min(max_slope * i,
                   ((ylen - 1) - min_slope * (xlen - 1)
                      + min_slope * i)) + 1);

        for j in range(int(minw),int(maxw)):

            if i + j == 0:
                cur[j] = abs(x[0] - y[0]);
            elif i == 0:
                cur[j] = cur[j-1] + msm_dist(y[j],y[j-1],x[0],c);
            elif j == minw:
                cur[j] = prev[j] +  msm_dist(x[i], y[0],x[i-1],c);
            else:
                cur[j] = min(prev[j-1] + abs(x[i] - y[j]),
                            prev[j] + msm_dist(x[i], x[i -1],y[j],c),
                            cur[j-1] + msm_dist(y[j], x[i], y[j-1],c));
        
        

    return cur[ylen-1];

def msm_ip(x,y,c,slope):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));
    xlen = len(x);
    ylen = len(y);

    min_slope = (1/slope) * (float(ylen)/float(xlen));
    max_slope = slope * (float(ylen)/float(xlen));


    for i in range(len(x)):
        temp = prev;
        prev = cur;
        cur = temp;
        minw =  np.ceil(max(min_slope * i,
                    ((ylen-1) - max_slope * (xlen - 1)
                        + max_slope * i)))
        maxw = np.floor(min(max_slope * i,
                   ((ylen - 1) - min_slope * (xlen - 1)
                      + min_slope * i)) + 1);

        for j in range(int(minw),int(maxw)):

            if i + j == 0:
                cur[j] = abs(x[0] - y[0]);
            elif i == 0:
                cur[j] = prev[j] + msm_dist(x[i],x[i-1],y[0],c);
            elif j == minw:
                cur[j] = cur[j-1] +  msm_dist(y[i], x[0],y[i-1],c);
            else:
                cur[j] = min(prev[j-1] + abs(x[i] - y[j]),
                            prev[j] + msm_dist(x[i], x[i -1],y[j],c),
                            cur[j-1] + msm_dist(y[j], x[i], y[j-1],c));
        
        

    return cur[ylen-1];

'''
# DON'T USE THIS IMPLEMENTATION
@jit(nopython=True)
def msm_dist(new, x, y, c):
    if ((x <= new) and (new <= y)) or ((y <= new) and (new <= x)):
        dist = c;
    else:
        dist = c + min(abs(new - x), abs(new - y))

    return dist;

@jit(nopython=True)
def msm_scb_numba(x,y,c,w):
    lenx = len(x)
    leny = len(y)

    acc_cost_mat = np.full((lenx, leny), np.inf)

    for i in range(lenx):
        wmin = max(0, i-w)
        wmax = min(leny-1, i+w)

        for j in range(wmin, wmax+1):
            if i + j == 0:
                acc_cost_mat[i, j] = abs(x[i] - y[j])
            elif i == 0:
                acc_cost_mat[i, j] = acc_cost_mat[i, j-1] + msm_dist(y[j], y[j-1], x[i], c)
            elif j == 0:
                acc_cost_mat[i, j] =  acc_cost_mat[i-1, j-1] + msm_dist(x[i], x[i-1], y[j], c)
            else:
                acc_cost_mat[i, j] = min(acc_cost_mat[i-1, j-1] + abs(x[i] - y[j]),
                                         acc_cost_mat[i, j-1] + msm_dist(y[j], y[j-1], x[i], c),
                                         acc_cost_mat[i-1, j-1] + msm_dist(x[i], x[i-1], y[j], c))

        return acc_cost_mat[lenx-1, leny-1]


'''
@jit(nopython=True)
def msm_scb_numba(x,y,c,w):

    xlen = len(x)
    ylen = len(y)
    cost = np.full((xlen, ylen), np.inf)

    cost[0][0] = abs(x[0] - y[0]);

    for i in range(1,len(x)):
        cost[i][0] = cost[i-1][0] + msm_dist(x[i],x[i-1],y[0],c)

    for i in range(1,len(y)):
        cost[0][i] = cost[0][i-1] + msm_dist(y[i], x[0],y[i-1],c)

    for i in range(1,xlen):
        for j in range(max(0, int(i-w)), min(ylen, int(i+w))):
            cost[i][j] = min(cost[i-1][j-1] + abs(x[i] - y[j]),
                            cost[i-1][j] + msm_dist(x[i], x[i -1],y[j],c),
                            cost[i][j-1] + msm_dist(y[j], x[i], y[j-1],c))

    return cost[xlen-1][ylen-1]


def msm_scb(x,y,c,w):

    xlen = len(x);
    ylen = len(y);

    prev = np.zeros(ylen);
    cur = np.zeros(ylen);


    for i in range(xlen):
        temp = prev;
        prev = cur;
        cur = temp;
        minw = max(0,i-w);
        maxw = min(len(y),i+w);

        for j in range(int(minw),int(maxw)):

            if i + j == 0:
                cur[j] = abs(x[0] - y[0]);
            elif i == 0:
                cur[j] = prev[j] + msm_dist(x[i],x[i-1],y[0],c);
            elif j == minw:
                cur[j] = cur[j-1] +  msm_dist(y[i], x[0],y[i-1],c);
            else:
                cur[j] = min(prev[j-1] + abs(x[i] - y[j]),
                            prev[j] + msm_dist(x[i], x[i -1],y[j],c),
                            cur[j-1] + msm_dist(y[j], x[i], y[j-1],c));
        
        

    return cur[ylen-1];

# Start of Sequence Weighted Alignment (SWALE)

def swale(x,y,p,r,epsilon,constraint=None,w=5, fast = True):

    r"""
    Sequence Weighted Alignment (SWALE) [1]_ is an :math:`\epsilon` based distance measure. 
    SWALE introduces a punishment and reward system that is not in Longest Common Subsequence (LCSS). 
    This is encapsulated the parameters p and r. 
    This allows the user to tailor how punishing mismatches are and how rewarding matches are. 
    This makes SWALE more detailing then LCSS as LCSS only records the number of matches 
    and Edit Distance on Real Sequences (EDR) as EDR does not rewards matches. 
    However, 
    this leaves the responsibility to the user to set three parameters to get meaningful results. 
    This can be very hard to do without extensive testing 
    and leaves the results of SWALE heavily variable to the parameters users choose.

    .. math:: 

        \begin{aligned}
            D^u(x,y,\epsilon,p,r) &= 0\\
            D^v(x,y,\epsilon,p,r) = D^h(x,y,\epsilon,p,r) & = 
            \begin{cases}
                \infty & \text{if $|x - y| \leq \epsilon$}\\
                p & \text{else}\\
            \end{cases}\\
            D^d(x,y,\epsilon,p,r)& =
            \begin{cases}
                r & \text{if $|x - y| \leq \epsilon$}\\
                \infty & \text{else}\\
            \end{cases}\\
            \pi(d_{n,m}) &= d_{n,m}
        \end{aligned}

    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param p: punishment of one mismatch
    :type p: float
    :param r: reward of one match
    :type r: float
    :param epsilon: the matching threshold
    :type epsilon: float
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: float, optional
    :param w: If ``constraint = "Sakoe-Chiba"`` , ``w`` is the largest temporal shift allowed between two time series; if  ``constraint = "Itakura"``, ``w`` is the slope of the "Itakura Parallelogram". Default to 100. 
    :type w: float, optional
    :param fast: whether or not to use fast (Numba) implementation,  default to ``True``.
    :type fast: bool, optional
    :return: SWALE distance

    **Example:**

    Input:

    .. code-block:: python

        X = np.array([3, 4, 76, 4, 5])
        Y = np.array([0, 3, 4])

        swale_distance = swale(X, Y, p = 1, r = 1, epsilon = 3)
        print(swale_distance)

    Output:

    .. code-block:: bash

        6.0

    **References:**

    .. [1] Michael D. Morse and Jignesh M. Patel. “An efficient and accurate methodfor  evaluating  time  series  similarity”.  In:Proceedings of the 2007 ACMSIGMOD international conference on Management of data569–580 (2007).
    """
    
    if constraint == "None":
        if fast == True:
            return swale_n_numba(x,y,p,r,epsilon)
        if fast == False:
            return swale_n(x,y,p,r,epsilon)
    elif constraint == "Sakoe-Chiba":
        if fast == True:
           return swale_scb_numba(x,y,p,r,epsilon,w)
           
        if fast == False:
            return swale_scb(x,y,p,r,epsilon,w);
    elif constraint == "Itakura":
        if fast == True:
            return swale_ip_numba(x,y,p,r,epsilon,w)
        if fast == False:
            return swale_ip(x,y,p,r,epsilon,w)
    else:
        return swale_n_numba(x,y,p,r,epsilon);


@jit(nopython=True)
def dswale(x,y,p,r,epsilon):

    x = dev(x)
    y = dev(y)
    
    df = np.zeros((len(x),len(y)))

    for i in range(len(y)):
        df[0][i] = i * p;
    for i in range(len(x)):
        df[i][0] = i * p;

    for i in range(1,len(x)):
        for j in range(1,len(y)):
            if (abs(x[i] - y[i]) <= epsilon):
                df[i][j] = df[i-1][j-1] + r;
            else:
               df[i][j] = max(df[i][j-1], df[i-1][j]) + p;

    return df[len(x)-1][len(y)-1];


@jit(nopython=True)
def wswale(x,y,p,r,epsilon, g):
    df = np.zeros((len(x),len(y)))

    xlen = len(x);
    ylen = len(y);

    weight_vector = [1 / (1 + np.exp(-g * (i - xlen / 2))) for i in
                         range(0, xlen)]

    for i in range(len(y)):
        df[0][i] = i * p;
    for i in range(len(x)):
        df[i][0] = i * p;

    for i in range(1,len(x)):
        for j in range(1,len(y)):
            if (abs(x[i] - y[i]) <= epsilon):
                df[i][j] = df[i-1][j-1] + r * weight_vector[min(abs(i-j),xlen)];
            else:
               df[i][j] = max(df[i][j-1], df[i-1][j]) + p * weight_vector[min(abs(i-j),xlen)];

    return df[len(x)-1][len(y)-1];

@jit(nopython=True)
def swale_n_numba(x,y,p,r,epsilon):
    df = np.zeros((len(x),len(y)))

    for i in range(len(y)):
        df[0][i] = i * p;
    for i in range(len(x)):
        df[i][0] = i * p;

    for i in range(1,len(x)):
        for j in range(1,len(y)):
            if (abs(x[i] - y[i]) <= epsilon):
                df[i][j] = df[i-1][j-1] + r;
            else:
               df[i][j] = max(df[i][j-1], df[i-1][j]) + p;

    return df[len(x)-1][len(y)-1];

def swale_n(x,y,p,r,epsilon):
    df = np.zeros((len(x),len(y)))

    for i in range(len(y)):
        df[0][i] = i * p;
    for i in range(len(x)):
        df[i][0] = i * p;

    for i in range(1,len(x)):
        for j in range(1,len(y)):
            if (abs(x[i] - y[i]) <= epsilon):
                df[i][j] = df[i-1][j-1] + r;
            else:
               df[i][j] = max(df[i][j-1], df[i-1][j]) + p;

    return df[len(x)-1][len(y)-1];

@jit(nopython=True)
def swale_ip_numba(x,y,p,r,epsilon,slope=5):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));
    xlen = len(x);
    ylen = len(y);

    min_slope = (1/slope) * (float(ylen)/float(xlen));
    max_slope = slope * (float(ylen)/float(xlen));


    for i in range(len(x)):
        temp = prev;
        prev = cur;
        cur = temp;
        minw =  np.ceil(max(min_slope * i,
                    ((ylen-1) - max_slope * (xlen - 1)
                        + max_slope * i)))
        maxw = np.floor(min(max_slope * i,
                   ((ylen - 1) - min_slope * (xlen - 1)
                      + min_slope * i)) + 1);
        for j in range(int(minw),int(maxw)):
            if i + j == 0:
                cur[j] = 0;
            elif i == 0:
                cur[j] = j * p;
            elif j == minw:
                cur[j] = i * p;
            else:
                if (abs(x[i] - y[i]) <= epsilon):
                    cur[j] = prev[j-1] + r;
                else:
                   cur[j] = max(prev[j], cur[j-1]) + p;
        

    return cur[len(y)-1];

def swale_ip(x,y,p,r,epsilon,slope=5):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));
    xlen = len(x);
    ylen = len(y);

    min_slope = (1/slope) * (float(ylen)/float(xlen));
    max_slope = slope * (float(ylen)/float(xlen));


    for i in range(len(x)):
        temp = prev;
        prev = cur;
        cur = temp;
        minw =  np.ceil(max(min_slope * i,
                    ((ylen-1) - max_slope * (xlen - 1)
                        + max_slope * i)))
        maxw = np.floor(min(max_slope * i,
                   ((ylen - 1) - min_slope * (xlen - 1)
                      + min_slope * i)) + 1);
        for j in range(int(minw),int(maxw)):
            if i + j == 0:
                cur[j] = 0;
            elif i == 0:
                cur[j] = j * p;
            elif j == minw:
                cur[j] = i * p;
            else:
                if (abs(x[i] - y[i]) <= epsilon):
                    cur[j] = prev[j-1] + r;
                else:
                   cur[j] = max(prev[j], cur[j-1]) + p;
        

    return cur[len(y)-1];


# @jit(nopython=True)
# def swale_scb_numba(x, y,p, r, m, w):

#     cur = np.full((1, len(y)), -np.inf)
#     prev = np.full((1, len(y)), -np.inf)

#     for i in range(len(x)):
#         minw = max(0, i - w)
#         maxw = min(len(y)-1, i + w)
#         prev = cur
#         cur = np.full((1, len(y)), -np.inf)

#         for j in range(int(minw), int(maxw)+1):
#             if i + j == 0:
#                 cur[j] = 0
#             elif i == 0:
#                 cur[j] = -j
#             elif j == 0:
#                 cur[j] = -i
#             else:
#                 if abs(x[i] - y[j]) <= m:
#                     s1 = 0
#                 else:
#                     s1 = -1

#                 cur[j] =  max(prev[j - 1] + s1, prev[j] - 1, cur[j - 1] - 1)

#     return 0 - cur[len(y) - 1]

'''
@jit(nopython=True)
def swale_scb_numba(x, y, p,r, m, w):

    cur = np.full((1, len(y)), -np.inf)
    prev = np.full((1, len(y)), -np.inf)

    for i in range(len(x)):
        minw = max(0, i - w)
        maxw = min(len(y)-1, i + w)
        prev = cur
        cur = np.full((1, len(y)), -np.inf)

        for j in range(int(minw), int(maxw)+1):
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = -j * p
            elif j == 0:
                cur[j] = -i * p
            else:
                if abs(x[i] - y[j]) <= m:
                    s1 = - r
                else:
                    s1 = - p

                cur[j] = max(max(prev[j - 1] + s1, prev[j] + s1), (cur[j - 1] + s1))

    return 0 - cur[len(y) - 1]
'''


@jit(nopython=True)
def swale_scb_numba(x,y,p,r,epsilon,w):

    cur = np.zeros(len(y))
    prev = np.zeros(len(y))


    for i in range(len(x)):

        prev = cur
        cur = np.zeros(len(y))
        minw = max(0,i-w)
        maxw = min(i+w,len(y)-1)

        for j in range(int(minw),int(maxw)+1):
            
            if i + j == 0:
                cur[j] = 0
            elif i == 0:
                cur[j] = j * p
            elif j == minw:
                cur[j] = i * p
            else:
                if (abs(x[i] - y[i]) <= epsilon):
                    cur[j] = prev[j-1] + r
                else:
                   cur[j] = min(prev[j], cur[j-1]) + p
        

    return cur[len(y)-1];


def swale_scb(x,y,p,r,epsilon,w):

    cur = np.zeros(len(y));
    prev = np.zeros(len(y));


    for i in range(1,len(x)):
        temp = prev;
        prev = cur;
        cur = temp;
        minw = max(0,i-w);
        maxw = min(i+w,len(y));

        for j in range(int(minw),int(maxw)):
            
            if i + j == 0:
                cur[j] = 0;
            elif i == 0:
                cur[j] = j * p;
            elif j == minw:
                cur[j] = i * p;
            else:
                if (abs(x[i] - y[i]) <= epsilon):
                    cur[j] = prev[j-1] + r;
                else:
                   cur[j] = max(prev[j], cur[j-1]) + p;
        

    return cur[len(y)-1];

# Start of Weighted Dynamic Time Warping (WDTW)

def wdtw(x, y, g, w, fast = True):

    r"""
    Weighted dynamic time warping (WDTW) [1]_ is a variation of DTW 
    that aims to give more importance to the shape similarity of two time series. 
    It does this through a weighted vector that penalizes the differences between i and j.

    .. math::

        \begin{equation*}
            D(x_i,y_j,w_{abs(i-j)}) = w_{abs(i-j)} * |x_i - y_j| \\
            \pi(d_{i,j}) = d_{i,j}
        \end{equation*}
        

    Note: w is a element of a weight vector whose elements are calculated as:

    .. math::

        \begin{equation*}
            w_i(g,|X|) = \frac{1}{1 + e^{-g * (i - \frac{|X|}{2})}}
        \end{equation*}

    where :math:`|X|` is the length of the time series X.

    :param X: a time series
    :type X: np.array
    :param Y: another time series
    :type Y: np.array
    :param g: a constant that determines the weight vector. (see the formula above)
    :type g: float
    :return: WDTW Distance

    **Example:**

    Input:

    .. code-block:: python

        X = np.array([3, 4, 76, 4, 5])
        Y = np.array([0, 3, 4])

        wdtw_distance = wdtw(X, Y, g = 0.25)
        print(wdtw_distance)

    Output:

    .. code-block:: bash

        1.0459354060018373

    **References:**

    .. [1] Young-Seon Jeong, Myong K. Jeong, and Olufemi A. Omitaomu. “Weighteddynamic time warping for time series classification”. In:Pattern Recognition4.2231 – 2240 (2011)

    """

    if fast == True:
        return wdtw_numba(x,y,g, w)
    if fast == False:
        return wdtw_n(x, y, g)

@jit(nopython=True)
def wdtw_numba(x, y, g, w):
    N = len(x)
    M = len(y)
    weight_vector = [1 / (1 + np.exp(-g * (i - len(x) / 2))) for i in
                         range(len(x))]
    D = np.full((N+1, M+1), np.inf)
    D[0, 0] = 0
    for i in range(1, N+1):
        for j in range(max(1, i-w), min(i+w, M)+1):
            cost = (x[i-1] - y[j-1])**2 * weight_vector[np.abs(i - j)]
            D[i, j] = cost + min(D[i-1,j],D[i-1,j-1],D[i,j-1])

    Dist = math.sqrt(D[N, M])

    return Dist

'''
@jit(nopython=True)
def wdtw_numba(x,y,g, w):


    xlen = len(x);
    ylen = len(y);

    weight_vector = [1 / (1 + np.exp(-g * (i - xlen / 2))) for i in
                         range(0, xlen)]
    
    pairwise_distances = np.zeros((xlen, ylen))

    for i, x_ in enumerate(y):
        for j, y_ in enumerate(x):
            pairwise_distances[i][j] = abs(x_ - y_)
    
    distances = distances = np.full((xlen,ylen),np.inf)

    for i in range(0, xlen):
            for j in range(0, ylen):
                if i + j == 0:
                    distances[0][0] = weight_vector[0] * pairwise_distances[0][0];
                elif i == 0:
                    distances[0][j] = distances[0][j - 1] + weight_vector[j] * \
                              pairwise_distances[0][j];
                elif j == 0:
                     distances[i][0] = distances[i - 1][0] + weight_vector[i] * \
                              pairwise_distances[i][0];
                else:
                    min_dist = min(distances[i][j - 1], distances[i - 1][j],
                                   distances[i - 1][j - 1])
                    distances[i][j] = (min_dist + weight_vector[np.abs(i - j)] *
                                   pairwise_distances[i][j])
    return distances[xlen - 1][ylen - 1]
'''

def wdtw_n(x,y,g):

    xlen = len(x);
    ylen = len(y);

    weight_vector = [1 / (1 + np.exp(-g * (i - xlen / 2))) for i in
                         range(0, xlen)]

    pairwise_distances = np.zeros((xlen, ylen))

    for i, x_ in enumerate(y):
        for j, y_ in enumerate(x):
            pairwise_distances[i][j] = abs(x_ - y_)

    distances = np.full((xlen,ylen),np.inf)

    for i in range(0, xlen):
            for j in range(0, ylen):
                if i + j == 0:
                    distances[0][0] = weight_vector[0] * pairwise_distances[0][0];
                elif i == 0:
                    distances[0][j] = distances[0][j - 1] + weight_vector[j] * \
                              pairwise_distances[0][j];
                elif j == 0:
                     distances[i][0] = distances[i - 1][0] + weight_vector[i] * \
                              pairwise_distances[i][0];
                else:
                    min_dist = min(distances[i][j - 1], distances[i - 1][j],
                                   distances[i - 1][j - 1])
                    distances[i][j] = (min_dist + weight_vector[np.abs(i - j)] *
                                   pairwise_distances[i][j])

    return distances[xlen - 1][ylen - 1]