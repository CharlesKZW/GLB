#%%
import numpy as np
from numba import jit
from math import floor, sqrt

from ElasticMeasures import ddtw, dtw, edr, msm, swale, twed
from ElasticMeasures import lcss
from ElasticMeasures import erp
from ElasticMeasures import wdtw

#%%


@jit(nopython = True)
def lb_keogh(y, x, XUE, XLE):
    leny = len(y)
    lb_dist = 0
    
    for i in range(leny):

        if y[i] > XUE[i]:
            lb_dist += (y[i] - XUE[i]) ** 2
        if y[i] < XLE[i]:
            lb_dist += (y[i] - XLE[i]) ** 2

    return sqrt(lb_dist)

@jit(nopython = True)
def lb_kim(y, x):
    lb_dist = max(abs(x[0] - y[0]),
                  abs(x[len(x)-1] - y[len(y)-1]),
                  abs(max(x)- max(y)),
                  abs(min(x)- min(y)))
    return lb_dist

@jit(nopython = True)
def lb_new(y, x, w):
    leny = len(y)
    lenx = len(x)
    lb_dist = (x[0]-y[0]) ** 2 + (x[lenx-1] - y[leny-1]) **2
    
    for i in range(1,leny-1):

        wmin = max(0, i - w)
        wmax = min(lenx - 1, i + w) 
        wx = np.array([i for i in x[wmin : wmax + 1]])
        Y = np.full(wx.shape[0], -y[i])
        
        diff = np.add(wx, Y)
        cost = min(np.square(diff))

        lb_dist = lb_dist + cost

    return sqrt(lb_dist)

@jit(nopython = True)
def envelope_cost(x, YUE, YLE): # note the returned value is distance squared
    lenx = len(x)
    x_dist = 0

    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += (x[i] - YUE[i]) **2
        if x[i] < YLE[i]:
            x_dist += (x[i] - YLE[i]) **2

    return x_dist

@jit(nopython = True)
def boundary_cost(x, y): # note the returned value is distance squared
    lenx = len(x)
    leny = len(y)
    fixed_dist = (x[0]-y[0]) **2 + (x[lenx-1] - y[leny-1])**2

    return fixed_dist


@jit(nopython = True)
def glb_dtw(y, x, XUE, XLE, YUE, YLE):
    leny = len(y)
    lenx = len(x)
    fixed_dist = (x[0]-y[0]) **2 + (x[lenx-1] - y[leny-1])**2

    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += (y[i] - XUE[i]) **2
        if y[i] < XLE[i]:
            y_dist += (y[i] - XLE[i]) **2

    x_dist = 0
    
    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += (x[i] - YUE[i]) **2
        if x[i] < YLE[i]:
            x_dist += (x[i] - YLE[i]) **2

    lb_dist = fixed_dist + max(x_dist, y_dist)


    return sqrt(lb_dist)


@jit(nopython = True)
def glb_dtw_QueryOnly(y, x, XUE, XLE): 
    # lb_keogh vs glb_dtw_QueryOnly: lb_keogh might capture boundary, 
    # but glb_dtw_QueryOnly strictly doesn't.
    leny = len(y)
    lenx = len(x)
    
    fixed_dist = 0

    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += (y[i] - XUE[i]) **2
        if y[i] < XLE[i]:
            y_dist += (y[i] - XLE[i]) **2

    x_dist = 0
    

    lb_dist = fixed_dist + max(x_dist, y_dist)


    return sqrt(lb_dist)


@jit(nopython = True)
def glb_dtw_QueryBoundary(y, x, XUE, XLE):
    leny = len(y)
    lenx = len(x)
    fixed_dist = (x[0]-y[0]) **2 + (x[lenx-1] - y[leny-1])**2
    

    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += (y[i] - XUE[i]) **2
        if y[i] < XLE[i]:
            y_dist += (y[i] - XLE[i]) **2

    x_dist = 0
    

    lb_dist = fixed_dist + max(x_dist, y_dist)


    return sqrt(lb_dist)

@jit(nopython = True)
def glb_dtw_QueryData(y, x, XUE, XLE, YUE, YLE):
    leny = len(y)
    lenx = len(x)
    
    
    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += (y[i] - XUE[i]) **2
        if y[i] < XLE[i]:
            y_dist += (y[i] - XLE[i]) **2

    x_dist = 0
    
    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += (x[i] - YUE[i]) **2
        if x[i] < YLE[i]:
            x_dist += (x[i] - YLE[i]) **2

    lb_dist = max(x_dist, y_dist)

    return sqrt(lb_dist)


@jit(nopython = True)
def lower_b(t,w):

  b = np.zeros(len(t))
  for i in range(len(t)):
    b[i] = min(t[max(0,i-w):min(len(t)-1,i+w)+1])
  
  return b

@jit(nopython = True)
def upper_b(t,w):

  b = np.zeros(len(t))
  for i in range(len(t)):
    b[i] = max(t[max(0,i-w):min(len(t)-1,i+w)+1])
  
  return b
    
@jit(nopython = True)
def lb_keogh_squared(x,u,l):
  sumd = 0
  for i in range(len(x)):
    if x[i] > u[i]:
      sumd += (x[i] - u[i]) ** 2
    if x[i] < l[i]:
      sumd += (x[i] - l[i]) ** 2
    
  return sumd 

@jit(nopython = True)
def lb_improved(x,y,w, YUE, YLE):
    h = []
    l = YLE
    u = YUE
    for i in range(len(y)):
        if x[i] <= l[i]:
            h.append(l[i])
        elif x[i] >= u[i]:
            h.append(u[i])
        else:
            h.append(x[i])

    upper_h = upper_b(h,w)
    lower_h = lower_b(h,w)

    return sqrt(lb_keogh_squared(x,u,l) + lb_keogh_squared(y,upper_h,lower_h))

@jit(nopython = True)
def lb_wdtw_A(y, x, g, XUE, XLE):
    
    leny = len(y)
    lenx = len(x)
    lb_dist = 0
    

    for i in range(leny):

        if y[i] > XUE[i]:
            lb_dist += (y[i] - XUE[i]) ** 2
        if y[i] < XLE[i]:
            lb_dist += (y[i] - XLE[i]) ** 2

    w0 = min([1 / (1 + np.exp(-g * (i - len(x) / 2))) for i in
                         range(0, len(x))])
    lb_dist = lb_dist * w0
    return sqrt(lb_dist)

@jit(nopython = True)
def lb_wdtw_B(y, x, g, XUE, XLE, YUE, YLE):
    leny = len(y)
    lenx = len(x)
    fixed_dist = (x[0]-y[0]) **2 + (x[lenx-1] - y[leny-1])**2

    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += (y[i] - XUE[i]) **2
        if y[i] < XLE[i]:
            y_dist += (y[i] - XLE[i]) **2

    x_dist = 0
    
    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += (x[i] - YUE[i]) **2
        if x[i] < YLE[i]:
            x_dist += (x[i] - YLE[i]) **2

    lb_dist = fixed_dist + max(x_dist, y_dist)

    w0 = min([1 / (1 + np.exp(-g * (i - len(x) / 2))) for i in
                         range(0, len(x))])
    
    lb_dist = lb_dist * w0
    return sqrt(lb_dist)


def dev(X):
    lenx = X.shape[1]
    dX = (2 * X[:, 1:lenx-1] + X[:, 2:lenx] - 3*X[:, 0:lenx-2])/4
    first_col = np.array([dX[:, 0]])
    last_col = np.array([dX[:, dX.shape[1]-1]])
    dX = np.concatenate((first_col.T, dX), axis = 1)
    dX = np.concatenate((dX, last_col.T), axis =1)
    return dX

@jit(nopython = True)
def lb_ddtw(y, x, XUE, XLE, YUE, YLE):

    leny = len(y)
    lenx = len(x)

    fixed_dist = (x[0]-y[0]) **2 + (x[lenx-1] - y[leny-1])**2

    y_dist = 0

    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += (y[i] - XUE[i]) **2
        if y[i] < XLE[i]:
            y_dist += (y[i] - XLE[i]) **2

    x_dist = 0
    
    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += (x[i] - YUE[i]) **2
        if x[i] < YLE[i]:
            x_dist += (x[i] - YLE[i]) **2

    lb_dist = fixed_dist + max(x_dist, y_dist)

    return sqrt(lb_dist)


@jit(nopython = True)
def lb_keogh_lcss(y, x,epsilon, XUE, XLE): # LB_Keogh_LCSS
    
    LE_lower = np.subtract(XLE,epsilon)
    UE_higher = np.add(XUE,epsilon)
    
    leny = len(y)
    
    sum = 0
    
    for i in range(leny):

        if y[i] >= LE_lower[i] and y[i] <= UE_higher[i]:
            sum += 1

    lb_dist = 1 - (sum/(min(len(x),len(y))))

    return lb_dist

@jit(nopython = True)
def lcss_subcost(x, y, epsilon):
    if abs(x-y) <= epsilon: 
        r = 1
    else:
        r = 0
    return r

@jit(nopython = True)
def glb_lcss(y, x, epsilon, XUE, XLE, YUE, YLE):
    lenx = len(x)
    leny = len(y)
    fixed_sum = lcss_subcost(x[0], y[0], epsilon) + lcss_subcost(x[lenx-1], y[leny-1], epsilon)
    
    XLE_lower = np.subtract(XLE,epsilon)
    XUE_higher = np.add(XUE,epsilon)
    
    y_reward = 0
    
    for i in range(1, leny-1):
    
        if y[i] >= XLE_lower[i] and y[i] <= XUE_higher[i]:
            y_reward += 1
    
    YLE_lower = np.subtract(YLE,epsilon)
    YUE_higher = np.add(YUE,epsilon)
    x_reward = 0
   
    for i in range(1, lenx-1):
       
        if x[i] >= YLE_lower[i] and x[i] <= YUE_higher[i]:
            x_reward += 1
    
    sum = fixed_sum + min(y_reward, x_reward)
    lb_dist = 1 - (sum/(min(len(x),len(y))))
    
    return lb_dist


@jit(nopython = True)
def lb_erp(x, y): 
    return abs(np.sum(x) - np.sum(y))

@jit(nopython = True)
def lb_kim_erp(y, x, m): 
    x_max = max(m, max(x))
    x_min = min(m, min(x))
    lb_dist = max(abs(x[0]-y[0]),
                  abs(x[len(x)-1] - y[len(y)-1]),
                  abs(x_max - max(y)),
                  abs(x_min - min(y)))
    return lb_dist

@jit(nopython = True)
def lb_keogh_erp(y, x, m, XUE, XLE): 
    leny = len(y)
    lenx = len(x)
    lb_dist = 0
    for i in range(leny):
        UE = max(m, XUE[i])
        LE = min(m, XLE[i])
        if y[i] > UE:
            lb_dist += (y[i] - UE) ** 2
        if y[i] < LE:
            lb_dist += (y[i] - LE) ** 2
    return sqrt(lb_dist)


@jit(nopython = True)
def glb_erp(y, x, m, XUE, XLE, YUE, YLE): # GLB_ERP
    lenx = len(x)
    leny = len(y)
    
    fixed_dist = min((x[lenx-1] - y[leny-1])**2, (x[lenx-1]-m)**2, (y[leny-1]- m)**2)

    y_dist = 0
    for i in range(1, leny-1):

    
        if y[i] > XUE[i]:
            y_dist += min((y[i]-XUE[i])**2, (y[i]-m)**2)
        elif y[i] < XLE[i]:
            y_dist += min((y[i]-XLE[i])**2, (y[i]-m)**2)
    
    x_dist = 0
    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += min((x[i]-YUE[i])**2, (x[i]-m)**2)
        elif x[i] < YLE[i]:
            x_dist += min((x[i]-YLE[i])**2, (x[i]-m)**2)

    lb_dist = fixed_dist + max(x_dist, y_dist)
    
    return sqrt(lb_dist)



@jit(nopython = True)
def edr_subcost(x, y, epsilon):
    if abs(x-y) <= epsilon:
        cost = 0
    else:
        cost = 1
    return cost

@jit(nopython = True)
def glb_edr(x, y, epsilon, XUE, XLE, YUE, YLE):
    lenx = len(x)
    leny = len(y)
    fixed_cost = 0 + min(edr_subcost(x[lenx-1], y[leny-1], epsilon), 1)
    y_dist = 0
    for i in range(1, leny-1):
        if y[i] > XUE[i]:
            y_dist += edr_subcost(y[i], XUE[i], epsilon)
        if y[i] < XLE[i]:
            y_dist += edr_subcost(y[i], XLE[i], epsilon)
    x_dist = 0
    for i in range(1, lenx-1):
        if x[i] > YUE[i]:
            x_dist += edr_subcost(x[i], YUE[i], epsilon)
        if x[i] < YLE[i]:
            x_dist += edr_subcost(x[i], YLE[i], epsilon)

    lb_dist = fixed_cost + max(x_dist, y_dist)

    return lb_dist

@jit(nopython = True)
def swale_subcost(x, y, epsilon, p , r):
    if abs(x-y) <= epsilon:
        cost = r
    else:
        cost = p

    return cost

@jit(nopython = True)
def glb_swale(x, y, p, r, epsilon, XUE, XLE, YUE, YLE): 
    lenx = len(x)
    leny = len(y)

    fixed_cost = 0 + min(swale_subcost(x[lenx-1], y[leny-1], epsilon, p, r), p)

    y_dist = 0
    for i in range(1, leny-1):
    
        if y[i] > XUE[i]:
            y_dist += swale_subcost(y[i], XUE[i], epsilon, p, r)
        if y[i] < XLE[i]:
            y_dist += swale_subcost(y[i], XLE[i], epsilon, p, r)
    
    x_dist = 0
    for i in range(1, lenx-1):
    
        if x[i] > YUE[i]:
            x_dist += swale_subcost(x[i], YUE[i], epsilon, p, r)
        if x[i] < YLE[i]:
            x_dist += swale_subcost(x[i], YLE[i], epsilon, p, r)

    lb_dist = fixed_cost + max(x_dist, y_dist)

    return lb_dist


@jit(nopython = True)
def lb_msm(y, x, c, XUE, XLE):

    lenx = len(x)
    leny = len(y)

    lb_dist = abs(x[0]-y[0])

    for i in range(1,leny):

        if y[i] > XUE[i] and y[i-1] >= y[i]:
            lb_dist += min(abs(y[i]-XUE[i]), c)
        if y[i] < XLE[i] and y[i-1] <= y[i]:
            lb_dist += min(abs(y[i]-XLE[i]), c)
    
    return lb_dist


@jit(nopython = True)
def glb_msm(x, y, c, XUE, XLE, YUE, YLE):
    leny = len(y)
    lenx = len(x)
    

    if y[leny-2]>=y[leny-1]>=x[lenx-1] or y[leny-2]<=y[leny-1]<=x[lenx-1] or x[lenx-2]<=x[lenx-1]<=y[leny-1] or x[lenx-2]>=x[lenx-1]>=y[leny-1]:
        fixed_dist = abs(x[0]-y[0]) + min(abs(x[lenx-1]-y[leny-1]), c)
    else:
        fixed_dist = abs(x[0]-y[0]) + min(
                                        abs(x[lenx-1]-y[leny-1]),
                                        c + abs(y[leny-1] - y[leny-2]),
                                        c + abs(x[lenx-1] - x[lenx-2]))

    y_dist = 0
    for i in range(1, leny-1):

        if y[i] > XUE[i]:
            y_dist += min(abs(y[i]-XUE[i]), c)
        if y[i] < XLE[i]:
            y_dist += min(abs(y[i]-XLE[i]), c)
        
    x_dist = 0
    for i in range(1, lenx-1):

        if x[i] > YUE[i]:
            x_dist += min(abs(x[i]-YUE[i]), c)
        if x[i] < YLE[i]:
            x_dist += min(abs(x[i]-YLE[i]), c)

    lb_dist = fixed_dist + max(y_dist, x_dist)

    return lb_dist

@jit(nopython = True)
def lb_msm_C(x, y, c, w):
    lenx = len(x)
    leny = len(y)

    lb_dist = abs(x[0]-y[0])

    for i in range(1,leny):

        wmin = max(1, i - w)
        wmax = min(lenx - 1, i + w) 

        UE = max(x[wmin : wmax + 1])
        LE = min(x[wmin : wmax + 1])

        if y[i] > UE and y[i-1] >= y[i]:
            lb_dist += min(abs(y[i]-UE), c)
        if y[i] < LE and y[i-1] <= y[i]:
            lb_dist += min(abs(y[i]-LE), c)
        if y[i] > max(UE, y[i-1]):
            lb_dist += min(abs(y[i]-UE), c+abs(y[i]-y[i-1]))
        if y[i] < min(LE, y[i-1]):
            lb_dist += min(abs(y[i]-LE), c+abs(y[i]-y[i-1]))
    
    return lb_dist


@jit(nopython = True)
def lb_twed(y, x, lamb, nu, XUE, XLE):
    leny = len(y)

    lb_dist = min((x[0] - y[0])**2, (x[0])**2 + nu + lamb, (y[0])**2 + nu + lamb)

    for i in range(1, leny):

        if y[i] > max(XUE[i], y[i-1]):
            lb_dist += min(nu, (y[i]- max(XUE[i], y[i-1]))**2)
        if y[i] < min(XLE[i], y[i-1]):
            lb_dist += min(nu, (y[i]- min(XLE[i], y[i-1]))**2)
        
    return lb_dist


@jit(nopython = True)
def glb_twed(x, y, lamb, XUE, XLE, YUE, YLE):
    leny = len(y)
    lenx = len(x)

    fixed_dist = abs(x[0]-y[0]) + min(
                                    abs(x[lenx-1]-y[leny-1]),
                                    abs(y[lenx-1]-y[lenx-2])+lamb,
                                    abs(x[lenx-1]-x[lenx-2])+lamb
                                    )

    
    y_dist = 0
    for i in range(1, leny-1):

        if y[i]>=XUE[i] and y[i-1]>=XUE[i]:
            y_dist += min((abs(y[i]-XUE[i]) + abs(y[i-1]-XUE[i])), (abs(y[i]-y[i-1])+lamb))
        if y[i]<=XLE[i] and y[i-1]<=XLE[i]:
            y_dist += min(abs(y[i]-XLE[i]) + abs(y[i-1]-XLE[i]), abs(y[i]-y[i-1])+lamb)

    x_dist = 0
    for i in range(1, lenx-1):

        if x[i]>=YUE[i] and x[i-1]>=YUE[i]:
            x_dist += min((abs(x[i]-YUE[i]) + abs(x[i-1]-YUE[i])), (abs(x[i]-y[i-1])+lamb))
        if y[i]<=YLE[i] and y[i-1]<=YLE[i]:
            x_dist += min(abs(x[i]-YLE[i]) + abs(x[i-1]-YLE[i]), abs(x[i]-x[i-1])+lamb)

    lb_dist = fixed_dist + max(y_dist, x_dist)

    return lb_dist


@jit(nopython = True)
def matlab_round(value):
    rounded_value = round(value)
    if value - rounded_value == 0.5:
        rounded_value +=1
    return rounded_value

@jit(nopython = True)     
def make_envelopes(X, w):
    num_rows = X.shape[0]
    num_columns = X.shape[1]
    upper_envelopes = np.zeros((num_rows, num_columns))
    lower_envelopes = np.zeros((num_rows, num_columns))
    

    for i in range(num_rows):
        for j in range(num_columns):
            wmin = max(0, j-w)
            wmax = min(num_columns-1, j+w)

            upper_envelopes[i, j] = max(X[i, wmin: wmax+1])
            lower_envelopes[i, j] = min(X[i, wmin: wmax+1])
            

    return upper_envelopes, lower_envelopes

@jit(nopython = True) 
def add_projection(x, y, YUE, YLE, keogh, window):
    H = []
    for i in range(len(y)):
        if x[i] <= YLE[i]:
            H.append(YLE[i])
        elif x[i] >= YUE[i]:
            H.append(YUE[i])
        else:
            H.append(x[i])
    
    HUE = upper_b(H, window)
    HLE = lower_b(H, window)

    lb_dist = np.sqrt(keogh **2 + lb_keogh_squared(y, HUE, HLE))

    return lb_dist
        

#%%
class Bounded1NN(object):

    r"""
    Class for implementing One Nearest Neighbors Search with Lower Bounding Measures

    :param metric: elastic measure to compute similarity, shoud be one of {"lcss", "erp"}
    :type metric: str
    :param constraint: the constraint to use, should be one of {``"Sakoe-Chiba"``, ``"Itakura"``}  or ``None``, default to ``None``.
    :type constraint: str, optional
    :param w: If`constraint = "Sakoe-Chiba"`,`w` is the largest temporal shift allowed between two time series. If `constraint = "Itakura"`, :code:`w` is the slope of the "Itakura Parallelogram". `w` defaults to 10000 (which means no constraint at all for time series with than 10000 elements if `w` is not customized.)
    :type w: float, optional
    :param epsilon: the matching threshold for Longest Common Subsequence (LCSS) measure
    :type epsilon: float, required only if :code:`constraint = "lcss"`.
    :param m: the gap variable
    :type m: float, required only if :code:`constraint = "erp"`.
    
    """

    def __init__(self, metric, lb = True, constraint = None, w = 10000, epsilon = 0.2, m = 0, g = 0.3, c =0.5, lamb =1, nu = 0.0001, timesx = None, timesy = None, p = 5, r = 1):
        self.metric = metric
        self.w = w
        self.epsilon = epsilon
        self.constraint = constraint
        self.m = m
        self.lb = lb
        self.g = g
        self.c = c
        self.timesx = timesx
        self.timesy = timesy
        self.lamb = lamb
        self.nu = nu
        self.p = p
        self.r = r

    def fit(self, X, Xlabel):
        r"""
        This function fits the 1NN classifier from the training dataset.

        :param X: training dataset
        :type X: np.array
        :param Xlabel: target values (labels)
        :type Xlabel: np.array
        :return: Fitted 1NN classifier
        """
        if self.metric == "ddtw":
            self.X = dev(X)
        else:
            self.X = X
        self.Xlabel = Xlabel

    def predict(self, Y):
        r"""
        Predic class lables for given dataset

        :param X: test samples
        :type X: np.array
        :return: Predicted class label for each data sample

        """
        if self.metric == "ddtw":
            Y = dev(Y)
        
        pruned = 0

        test_class = np.zeros(Y.shape[0])
        window = self.w * Y.shape[1]
        window = matlab_round(window)
        
        if self.lb == True: 

            if self.metric in ["GLB_DTW", "ddtw", "GLB_ERP",  "GLB_LCSS", "GLB_EDR", "GLB_SWALE", "GLB_MSM", "GLB_TWED", "Breakdown_QueryData"]:
                self.XUE, self.XLE = make_envelopes(self.X, window)
            
            if self.metric in ["LB_Keogh", "GLB_DTW", "ddtw", "LB_Improved", "Cas_Keogh_New", "Cas_Keogh_Improved","Cas_Keogh_GLB", "GLB_ERP", "LB_Keogh_ERP", "LB_Keogh_LCSS", "GLB_LCSS", "GLB_EDR", "GLB_SWALE", "wdtw_A", "wdtw_B",  "LB_MSM", "GLB_MSM", "LB_TWED", "GLB_TWED", "Breakdown_QueryData", "Breakdown_QueryOnly", "Breakdown_QueryBoundary"]:
                self.YUE, self.YLE = make_envelopes(Y, window)

            if self.metric == "Cas_Keogh_GLB":
                self.XUEC = np.zeros((self.X.shape[0], self.X.shape[1]))
                self.XLEC = np.zeros((self.X.shape[0], self.X.shape[1]))

                assert self.XUEC.shape == self.X.shape
                assert self.XLEC.shape == self.X.shape


            for idx_y, y in enumerate(Y):

                best_so_far = float('inf')

            
                lb_list = np.zeros(self.X.shape[0])
                for idx_x, x in enumerate(self.X):
                    if self.metric == "LB_Keogh":
                        lb_dist = lb_keogh(x, y, self.YUE[idx_y], self.YLE[idx_y])
                    if self.metric == "GLB_DTW":
                        lb_dist = glb_dtw(x, y, self.YUE[idx_y], self.YLE[idx_y],  self.XUE[idx_x], self.XLE[idx_x])
                    if self.metric == "ddtw":
                        lb_dist = glb_dtw(x, y, self.YUE[idx_y], self.YLE[idx_y],  self.XUE[idx_x], self.XLE[idx_x])
                    if self.metric == "LB_Kim":
                        lb_dist = lb_kim(x, y)
                    if self.metric == "LB_New":
                        lb_dist = lb_new(x, y, window)
                    if self.metric == "LB_Improved":
                        lb_dist = lb_improved(x, y, window, self.YUE[idx_y], self.YLE[idx_y])
                    if self.metric ==  "Cas_Keogh_New":
                        lb_dist = lb_keogh(x, y, self.YUE[idx_y], self.YLE[idx_y])
                    if self.metric ==  "Cas_Keogh_Improved":
                        lb_dist = lb_keogh(x, y, self.YUE[idx_y], self.YLE[idx_y])
                        
                    if self.metric ==  "Cas_Keogh_GLB":
                        
                        lb_dist = sqrt(envelope_cost(x, self.YUE[idx_y], self.YLE[idx_y]))


                    if self.metric == "Breakdown_QueryData":
                        lb_dist = glb_dtw_QueryData(x, y, self.YUE[idx_y], self.YLE[idx_y],  self.XUE[idx_x], self.XLE[idx_x])

                    if self.metric == "Breakdown_QueryOnly":
                        lb_dist = glb_dtw_QueryOnly(x, y, self.YUE[idx_y], self.YLE[idx_y])
                    
                    if self.metric == "Breakdown_QueryBoundary":
                        lb_dist = glb_dtw_QueryBoundary(x, y, self.YUE[idx_y], self.YLE[idx_y])

                    if self.metric == "LB_ERP":
                        lb_dist = lb_erp(x, y)
                    if self.metric == "GLB_ERP":
                        lb_dist = glb_erp(x, y, self.m, self.YUE[idx_y], self.YLE[idx_y],  self.XUE[idx_x], self.XLE[idx_x])
                    if self.metric == "LB_Keogh_ERP":
                        lb_dist = lb_keogh_erp(x, y, self.m, self.YUE[idx_y], self.YLE[idx_y])
                    if self.metric == "LB_Kim_ERP":
                        lb_dist = lb_kim_erp(x, y, self.m)
                    if self.metric == "LB_Keogh_LCSS":
                        lb_dist = lb_keogh_lcss(x, y, self.epsilon, self.YUE[idx_y], self.YLE[idx_y])
                    if self.metric == "GLB_LCSS":
                        lb_dist = glb_lcss(x, y, self.epsilon, self.YUE[idx_y], self.YLE[idx_y], self.XUE[idx_x], self.XLE[idx_x])
                    if self.metric == "GLB_EDR":
                        lb_dist = glb_edr(x, y, self.epsilon, self.XUE[idx_x], self.XLE[idx_x], self.YUE[idx_y], self.YLE[idx_y])
                    if self.metric == "GLB_SWALE":
                        lb_dist = glb_swale(x, y, self.p, self.r, self.epsilon, self.XUE[idx_x], self.XLE[idx_x], self.YUE[idx_y], self.YLE[idx_y])
                    if self.metric == "wdtw_A":
                        lb_dist = lb_wdtw_A(x, y, self.g, self.YUE[idx_y], self.YLE[idx_y])
                    if self.metric == "wdtw_B":
                        lb_dist = lb_wdtw_B(x, y, self.g, self.YUE[idx_y], self.YLE[idx_y],  self.XUE[idx_x], self.XLE[idx_x])
                    if self.metric == "LB_MSM":
                        lb_dist = lb_msm(x, y, self.c, self.YUE[idx_y], self.YLE[idx_y])
                    if self.metric == "GLB_MSM":
                        lb_dist = glb_msm(x, y, self.c, self.XUE[idx_x], self.XLE[idx_x], self.YUE[idx_y], self.YLE[idx_y])
                    if self.metric == "msm_C":
                        lb_dist = lb_msm_C(x, y, self.c, window)
                    if self.metric == "LB_TWED":
                        lb_dist = lb_twed(x, y, self.lamb, self.nu, self.YUE[idx_y], self.YLE[idx_y])
                    if self.metric == "GLB_TWED":
                        lb_dist = glb_twed(x, y, self.lamb, self.XUE[idx_x], self.XLE[idx_x], self.YUE[idx_y], self.YLE[idx_y])
                

                    lb_list[idx_x] = lb_dist
                    
                    
                

                ordering = np.argsort(lb_list)

                self.X = self.X[ordering]
                self.Xlabel = self.Xlabel[ordering]
                lb_list = lb_list[ordering]
                if self.metric in ["GLB_DTW", "ddtw", "GLB_ERP",  "GLB_LCSS", "GLB_EDR", "GLB_SWALE", "GLB_TWED", "Breakdown_QueryData"]:
                    self.XUE = self.XUE[ordering]
                    self.XLE = self.XLE[ordering]

                if self.metric == "Cas_Keogh_GLB":
                    self.XUEC = self.XUEC[ordering]
                    self.XLEC = self.XLEC[ordering]

            
            
                for idx_x, x in enumerate(self.X):

                    lb_dist = lb_list[idx_x]

                    if self.metric == "Cas_Keogh_New" and lb_dist < best_so_far:

                        lb_dist = lb_new(x, y, window)
                    
                    if self.metric == "Cas_Keogh_Improved" and lb_dist < best_so_far:

                        lb_dist = add_projection(x, y, self.YUE[idx_y], self.YLE[idx_y], lb_dist, window)
                

                    if self.metric == "Cas_Keogh_GLB" and lb_dist < best_so_far:

                        if sum(self.XUEC[idx_x]) == 0 and sum(self.XLEC[idx_x]) == 0:

                            self.XUEC[idx_x], self.XLEC[idx_x] = make_envelopes(np.array([x]), window)

                        UE = self.XUEC[idx_x]

                        LE = self.XLEC[idx_x]

                        lb_dist = max(lb_dist, sqrt(envelope_cost(y, np.squeeze(UE), np.squeeze(LE)))) + boundary_cost(x, y)

                    if lb_dist < best_so_far:

                        #change actual algo
                        if self.metric == "LB_Keogh" or self.metric == "GLB_DTW" or self.metric == "LB_Kim" or self.metric == "LB_Improved" or self.metric == "LB_New" or self.metric == "Cas_Keogh_New" or self.metric == "Cas_Keogh_Improved" or self.metric == "Cas_Keogh_GLB" or self.metric == "Breakdown_QueryData" or self.metric == "Breakdown_QueryOnly" or self.metric == "Breakdown_QueryBoundary": 
                            actual_dist = dtw(x, y, window, self.constraint)

                        if self.metric == "ddtw":
                            actual_dist = dtw(x, y, window, self.constraint)

                        if self.metric == "LB_Keogh_LCSS" or self.metric == "GLB_LCSS":
                            actual_dist = lcss(x, y, self.epsilon, window, self.constraint)

                        if self.metric == "GLB_EDR":
                            actual_dist = edr(x, y, self.epsilon, window, self.constraint)

                        if self.metric == "GLB_SWALE":
                            actual_dist = swale(x, y, self.p, self.r, self.epsilon, self.constraint, window)
                        
                        if self.metric == "LB_ERP" or self.metric == "GLB_ERP" or self.metric == "LB_Keogh_ERP" or self.metric == "LB_Kim_ERP":
                            actual_dist = erp(x, y, self.m, self.constraint, window)

                        if self.metric == "wdtw_A" or self.metric == "wdtw_B":
                            actual_dist = wdtw(x, y, self.g, window)
                        
                        if self.metric == "LB_MSM" or self.metric == "GLB_MSM" or self.metric == "msm_C":
                            actual_dist = msm(x, y, self.c, self.constraint, window)

                        if self.metric == "LB_TWED" or self.metric == "GLB_TWED" :
                            if self.timesx == None:
                                timestampx = np.array([i for i in range(len(x))])
                            else:
                                timestampx = self.timesx

                            if self.timesy == None:
                                timestampy = np.array([i for i in range(len(y))])
                            else:
                                timestampy = self.timesy
                            actual_dist = twed(x, timestampx, y, timestampy, self.lamb, self.nu, self.constraint, window)

                        if actual_dist < best_so_far:
                            best_so_far = actual_dist
                            test_class[idx_y] = self.Xlabel[idx_x]

                    
                    if lb_dist > best_so_far:
                        pruned += 1
                    
                    
        if self.lb == False:

            for idx_y, y in enumerate(Y):

                best_so_far = float('inf')

                for idx_x, x in enumerate(self.X):
 
                    if self.metric == "LB_Keogh" or self.metric == "GLB_DTW" or self.metric == "LB_Kim" or self.metric == "LB_Improved" or self.metric == "LB_New" or self.metric == "Cas_Keogh_New" or self.metric == "Cas_Keogh_Improved" or self.metric == "Cas_Keogh_GLB" or self.metric == "Breakdown_QueryData" or self.metric == "Breakdown_QueryOnly" or self.metric == "Breakdown_QueryBoundary":
                        actual_dist = dtw(x, y, window, self.constraint)

                    if self.metric == "ddtw":
                        actual_dist = dtw(x, y, window, self.constraint)

                    if self.metric == "LB_Keogh_LCSS" or self.metric == "GLB_LCSS":
                        actual_dist = lcss(x, y, self.epsilon, window, self.constraint)
                    
                    if self.metric == "GLB_EDR":
                        actual_dist = edr(x, y, self.epsilon, window, self.constraint)
                    
                    if self.metric == "GLB_SWALE":
                        actual_dist = swale(x, y, self.p, self.r, self.epsilon, self.constraint, window)
                    
                    if self.metric == "LB_ERP" or self.metric == "GLB_ERP" or self.metric == "LB_Keogh_ERP" or self.metric == "LB_Kim_ERP":
                        actual_dist = erp(x, y, self.m, self.constraint, window)
                    
                    if self.metric == "wdtw_A" or self.metric == "wdtw_B":
                        actual_dist = wdtw(x, y, self.g, window)

                    if self.metric == "LB_MSM" or self.metric == "GLB_MSM" or self.metric == "msm_C":
                        actual_dist = msm(x, y, self.c, self.constraint, window)

                    if self.metric == "LB_TWED" or self.metric == "GLB_TWED":
                        if self.timesx == None:
                            timestampx = np.array([i for i in range(len(x))])
                        else:
                            timestampx = self.timesx

                        if self.timesy == None:
                            timestampy = np.array([i for i in range(len(y))])
                        else:
                            timestampy = self.timesy
                        actual_dist = twed(x, timestampx, y, timestampy, self.lamb, self.nu, self.constraint, window)
                    

                    if actual_dist < best_so_far:
                        best_so_far = actual_dist
                        test_class[idx_y] = self.Xlabel[idx_x]
                    
                
        pruning_power = pruned / (Y.shape[0] * self.X.shape[0])
            
        
        return test_class, pruning_power
        
        
