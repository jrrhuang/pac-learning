import random
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm
import collections
import networkx as nx
from operator import add
from scipy.optimize import fsolve, linprog, minimize
from scipy.special import lambertw
import cvxpy as cp


# An algorithm takes as an input a sequence of requests.
# It returns its full history of execution, i.e. a list of length len(requests),
# with the i-th element representing the cost to serve the (i+1)-th request 

RHO = 1.1596  # default value of rho
RHO_LIMIT = 1.1596

POWER_CONSUMPTIONS = [1., 0.47, 0.105, 0]
WAKE_UP_COSTS = [0, 0.12, 0.33, 1.] # assume all are interesting at some point

# Rent and buy costs
R = 1
# B = 40
B = 80

# PAC parameter
# delta
d = 0.01
d_ppp = 0.4

#POWER_CONSUMPTIONS = [1., 0]
#WAKE_UP_COSTS = [0, 1.] # assume all are interesting at some point


def inverse(f, delta=1/1024.):
    def f_1(y):
        return binary_search(f, y, 0, 1, delta)
    return f_1

def binary_search(f, y, lo, hi, delta):
    while lo <= hi:
        x = (lo + hi) / 2
        if f(x) < y:
            lo = x + delta
        elif f(x) > y:
            hi = x - delta
        else:
            return x;
    return hi if (f(hi) - y < y - f(lo)) else lo

# #################### Predictions start here ####################

# Each prediction returns a list of len(requests) integers, which predicts the next request time

# Synthetic predictions.
# Ground truth + add a random noise following a normal distribution  
def PredRandom(requests,sigma=0.5):
  # noise = np.round(np.random.normal(0,sigma,len(requests)))
  noise = np.random.normal(0,sigma,len(requests))
  return  [max(1,x) for x in requests+noise]

def PPP(requests, sigma=0.5, d=d_ppp):
  # noise = np.round(np.random.normal(0,sigma,len(requests)))
  noise = np.random.normal(0,sigma,len(requests))
  return [max(1,requests[i]+noise[i]) if np.random.rand() < d else requests[i] for i in range(len(requests))]


def Buy(requests, pred=[], sigmas=[], B=B):
  return [0 for x in requests]
  

def PredHelmbold(requests, ETA=4.0):
  ALPHA = 0.08
  LALPHA = np.log(1.0 - ALPHA)
  MAX_TIMEOUT = (WAKE_UP_COSTS[-1] - WAKE_UP_COSTS[-2]) / POWER_CONSUMPTIONS[-2]
  N = 25
  TIMEOUT = np.array([MAX_TIMEOUT / (1.15 ** i) for i in range(N)])
  weight = np.ones(N) / N
  predictions = []
  for request in requests:
    weight = weight / np.sum(weight)
    prediction = np.sum(TIMEOUT * weight)
    predictions.append(prediction)
    loss = np.abs(TIMEOUT - request) / MAX_TIMEOUT
    weight = weight * np.exp(-ETA * loss)
    alphas = np.exp(LALPHA * loss)
    pool = np.sum(weight * (1 - alphas))
    weight = weight * alphas + pool / N
  return predictions


# #################### Predictions end here ####################


# #################### Two-states Algorithms start here ####################
# each algorithm returns buying time, with a buying time > 1000 means it neve buys

def OPT(requests, pred=[]):
  return [1 if x>B else x+1000 for x in requests]

def Rent(requests, pred=[]):
  return [ x+1000 for x in requests]

# e/e-1 algorithm: probability density = e^(x/B)/(e-1) so CDF = (e^(x/B)-1)/(e-1)
def ClassicRandom(requests, pred=[]):
  res = [0] * len(requests)
  for t, req in enumerate(requests):
    buy = B*np.log(1+np.random.rand()*(np.expm1(1)))
    res[t] = buy
  return res
  
# 2-competitive algorithm
def ClassicDet(requests, pred=[]):
  res = [0] * len(requests)
  for t, req in enumerate(requests):
    res[t] = B
  return res

# Kumar Purohit Svitkina algorithm. 
# rho parameter: consistency
# def KumarRandom(requests, pred, sigma, rho = 1.4):  
#   lambd = inverse(lambda x : x/(1-np.exp(-x))) (rho)
#   res = [0] * len(requests)
#   for t, req in enumerate(requests):
#     k = lambd if pred[t] >= B else B/lambd
#     buy = np.log(1+np.random.rand()*(np.expm1(k)))
#     res[t] = buy
#   return res

def KumarRandom(requests, pred, lmbda=0.2):
  res = [0] * len(requests)
  for t, req in enumerate(requests):
    if pred[t] >= B:
      k = int(np.floor(lmbda*B))
      prob = [((B-1)/B)**(k-i)/(B*(1-(1-1/B)**k)) for i in range(1,k+1)]
      res[t] = np.random.choice(np.arange(1,k+1), p=prob)
    else:
      ell = int(np.ceil(B/lmbda))
      prob = [((B-1)/B)**(ell-i)/(B*(1-(1-1/B)**ell)) for i in range(1,ell+1)]
      res[t] = np.random.choice(np.arange(1,ell+1), p=prob)
  return res

def Kumar1(requests, pred, sigmas):
  return KumarRandom(requests, pred, lmbda=0.2)

def Kumar2(requests, pred, sigmas):
  return KumarRandom(requests, pred, lmbda=0.5)

def Kumar3(requests, pred, sigmas):
  return KumarRandom(requests, pred, lmbda=0.8)

def Meta(requests, pred, sigmas, d):
  res = [0] * len(requests)
  z = norm.ppf(1-d/2)
  for t, req in enumerate(requests):
    if t % 1000 == 0:
      print(t)
    l, u = max(1,pred[t] - z*sigmas[t]), pred[t] + z*sigmas[t]
    def fun_cr(x):
      return d / (1-np.exp(-(x-1/B))) + (1-d)*x/(1-np.exp(-x))*(1+(u-l)/(2*min(l,B)))
    r = minimize(fun_cr, 0.5, bounds=((1/B,1),))
    lmbda = r.x[0]
    if pred[t] >= B:
      k = int(np.floor(lmbda*B))
      prob = [((B-1)/B)**(k-i)/(B*(1-(1-1/B)**k)) for i in range(1,k+1)]
      res[t] = np.random.choice(np.arange(1,k+1), p=prob)
    else:
      ell = int(np.ceil(B/lmbda))
      prob = [((B-1)/B)**(ell-i)/(B*(1-(1-1/B)**ell)) for i in range(1,ell+1)]
      res[t] = np.random.choice(np.arange(1,ell+1), p=prob)
  return res

def Meta1(requests, pred, sigmas):
  return Meta(requests, pred, sigmas, 0.1)

def Meta2(requests, pred, sigmas):
  return Meta(requests, pred, sigmas, 0.01)
  
# either buys at time 0 or never
def FTP(requests, pred):
  res = [0] * len(requests)
  for t, req in enumerate(requests):
     res[t] = 1 if pred[t]>=B else requests[t]+1000
  return res

# deterministic algorithm from Angelopoulos et al
def AngelopoulosDet(requests, pred, rho=RHO_LIMIT):
  res = [0] * len(requests)
  for t, req in enumerate(requests):
    buy = 1 if pred[t]<=1 else rho-1
    res[t] = buy
  return res

def PACPredict(requests, pred, sigmas, d):
  res = [0] * len(requests)
  z = norm.ppf(1-d/2)
  for t, req in enumerate(requests):
    l, u = max(1,pred[t] - z*sigmas[t]), pred[t] + z*sigmas[t]
    if u < B:
      res[t] = B
    elif l > B:
      if min(d+np.sqrt(d*(1-d)),1)+max(1-d+np.sqrt(d*(1-d)),1) <= d+u/B:
        res[t] = min(B*np.sqrt(d/(1-d)),B)
      else:
        res[t] = u
    else:
      if min(d+np.sqrt(d*(1-d)*B/l),1)+max((1-d)*B/l+np.sqrt(d*(1-d)*B/l),B/l) >= 2 and d+u/B >= 2:
        res[t] = B
      elif min(d+np.sqrt(d*(1-d)*B/l),1)+max((1-d)*B/l+np.sqrt(d*(1-d)*B/l),B/l) <= d+u/B:
        res[t] = l*min(np.sqrt(d*B/((1-d)*l)),1)
      else:
        res[t] = u
  return res

def PACPredictRand(requests, pred, sigmas, d):
  # NOTE: assumes skiing days lower bounded by 1
  result = [0] * len(requests)
  z = norm.ppf(1-d/2)
  for t, req in enumerate(requests):
    if t % 1000 == 0:
      print(t)
    l, u = max(1,pred[t] - z*sigmas[t]), pred[t] + z*sigmas[t]
    l, u = pred[t] - z*sigmas[t], pred[t] + z*sigmas[t]
    if u < B:
      c = np.zeros((B+2,))
      c[:2] = np.array([1-d,d])
      A_ub = np.zeros((B+1,B+2))
      for T_idx in range(B):
        T = T_idx + 1
        if l <= T and T <= u:
          A_ub[T_idx,0] = -min(T,B)
        else:
          A_ub[T_idx,1] = -min(T,B)
        A_ub[T_idx,2:] = np.array([B+t-1 if t <= T else T for t in range(1,B+1)])
      A_ub[B,:2] = np.array([1,-1])
      b_ub = np.zeros((B+1,))

      bounds = [(1,None)]*2 + [(0,None)]*B
      A_eq = [[0]*2+[1]*B]
      b_eq = [1]
    else:
      c = np.zeros((B+3,))
      c[:2] = np.array([1-d,d])
      A_ub = np.zeros((B+2,B+3))
      for T_idx in range(B+1):
        if T_idx < B-1:
          T = T_idx + 1
        elif T_idx == B-1:
          T = u
        else:
          T = u+1
        if l <= T and T <= u:
          A_ub[T_idx,0] = -min(T,B)
        else:
          A_ub[T_idx,1] = -min(T,B)
        lst = list(range(1,B+1))+[u+1]
        A_ub[T_idx,2:B+3] = np.array([B+t-1 if t <= T else T for t in lst])
      A_ub[B+1,:2] = np.array([1,-1])
      b_ub = np.zeros((B+2,))

      bounds = [(1,None)]*2 + [(0,None)]*(B+1)
      A_eq = [[0]*2+[1]*(B+1)]
      b_eq = [1]

    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
    dist = np.array(res.x[2:])
    if u < B:
      dist[dist < 0] = 0
      action = np.random.choice(range(B), p=dist) + 1
    else:
      dist[dist < 0] = 0
      action = np.random.choice(range(B+1), p=dist)
      if action == B:
        action = u+1
      else:
        action += 1
    result[t] = action
  return result


# def PACPredictRand(requests, pred, sigmas, d):
#   # NOTE: assumes skiing days lower bounded by 1
#   MAX_REQUEST = int(np.ceil(max(requests)))
#   result = [0] * len(requests)
#   z = norm.ppf(1-d/2)
#   for t, req in enumerate(requests):
#     if t % 1000 == 0:
#       print(t)
#     l, u = max(1,pred[t] - z*sigmas[t]), pred[t] + z*sigmas[t]
#     l, u = pred[t] - z*sigmas[t], pred[t] + z*sigmas[t]
    
#     c = np.zeros((MAX_REQUEST+2,))
#     c[:2] = np.array([1-d,d])
#     A_ub = np.zeros((MAX_REQUEST+1,MAX_REQUEST+2))
#     for T_idx in range(MAX_REQUEST):
#       T = T_idx + 1
#       if l <= T and T <= u:
#         A_ub[T_idx,0] = -min(T,B)
#       else:
#         A_ub[T_idx,1] = -min(T,B)
#       A_ub[T_idx,2:] = np.array([B+t-1 if t <= T else T for t in range(1,MAX_REQUEST+1)])
#     A_ub[MAX_REQUEST,:2] = np.array([1,-1])
#     b_ub = np.zeros((MAX_REQUEST+1,))

#     bounds = [(1,None)]*2 + [(0,None)]*MAX_REQUEST
#     A_eq = [[0]*2+[1]*MAX_REQUEST]
#     b_eq = [1]

#     res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)
#     dist = np.array(res.x[2:])
#     dist[dist < 0] = 0
#     action = np.random.choice(range(MAX_REQUEST), p=dist) + 1
#     result[t] = action
#   return result


def PACPredictRandCVX(requests, pred, sigmas, d):
  # NOTE: assumes skiing days lower bounded by 1
  result = [0] * len(requests)
  z = norm.ppf(1-d/2)
  for t, req in enumerate(requests):
    if t % 1000 == 0:
      print(t)
    l, u = pred[t] - z*sigmas[t], pred[t] + z*sigmas[t]
    if u < B:
      c = np.zeros((B+2,))
      c[:2] = np.array([1-d,d])
      A_ub = np.zeros((B+1,B+2))
      for T_idx in range(B):
        T = T_idx + 1
        if l <= T and T <= u:
          A_ub[T_idx,0] = -min(T,B)
        else:
          A_ub[T_idx,1] = -min(T,B)
        A_ub[T_idx,2:] = np.array([B+t-1 if t <= T else T for t in range(1,B+1)])
      A_ub[B,:2] = np.array([1,-1])
      b_ub = np.zeros((B+1,))

      A_eq = np.ones((B+2,))
      A_eq[:2] = np.zeros((2,))
      b_eq = np.array([1])

      A_bnd = np.eye(B+2)
      b_bnd = np.zeros((B+2,))
      b_bnd[:2] = np.array([1,1])

      x = cp.Variable(B+2)
    else:
      c = np.zeros((B+3,))
      c[:2] = np.array([1-d,d])
      A_ub = np.zeros((B+2,B+3))
      for T_idx in range(B+1):
        if T_idx < B-1:
          T = T_idx + 1
        elif T_idx == B-1:
          T = u
        else:
          T = u+1
        if l <= T and T <= u:
          A_ub[T_idx,0] = -min(T,B)
        else:
          A_ub[T_idx,1] = -min(T,B)
        lst = list(range(1,B+1))+[u+1]
        A_ub[T_idx,2:B+3] = np.array([B+t-1 if t <= T else T for t in lst])
      A_ub[B+1,:2] = np.array([1,-1])
      b_ub = np.zeros((B+2,))

      A_eq = np.array([[0]*2+[1]*(B+1)])
      b_eq = np.array([1])

      A_bnd = np.eye(B+3)
      b_bnd = np.zeros((B+3,))
      b_bnd[:2] = np.array([1,1])

      x = cp.Variable(B+3)

    prob = cp.Problem(cp.Minimize(c.T@x),
                      [A_ub @ x <= b_ub,
                        A_eq @ x == b_eq,
                        A_bnd @ x >= b_bnd])
    prob.solve()
    dist = x.value[2:]
    if u < B:
      dist[dist < 0] = 0
      action = np.random.choice(range(B), p=dist) + 1
    else:
      dist[dist < 0] = 0
      action = np.random.choice(range(B+1), p=dist)
      if action == B:
        action = u+1
      else:
        action += 1
    result[t] = action
  return result

def PACPredict1(requests, pred, sigmas):
  return PACPredict(requests, pred, sigmas, 0.1)

def PACPredict2(requests, pred, sigmas):
  return PACPredict(requests, pred, sigmas, 0.05)

def PACPredict3(requests, pred, sigmas):
  return PACPredict(requests, pred, sigmas, 0.01)

def PACPredictRand1(requests, pred, sigmas):
  return PACPredictRand(requests, pred, sigmas, 0.1)

def PACPredictRand2(requests, pred, sigmas):
  return PACPredictRand(requests, pred, sigmas, 0.01)

def PPPAlgorithm(requests, pred, d=d_ppp):
  res = [0] * len(requests)
  for t, req in enumerate(requests):
    P = pred[t]
    if d < 0.5:
      cr1 = 1+2*np.sqrt(d*(1-d))
    else:
      cr1 = 2
    cr2 = d + P / B

    if P <= B:
      res[t] = B
    elif P <= (np.sqrt(5)+1)/2*B:
      if cr1 <= cr2:
        res[t] = B * min(np.sqrt(d/(1-d)),1)
      else:
        res[t] = P
    else:
      res[t] = B * min(np.sqrt(d/(1-d)),1)
  return res

  
  
# #################### Function computing the CDF of our algorithm start here ####################


# buying time at time 0 in many cases
def g0(mu,tp):
  return tp*mu
  
# density of two parts of the CDF
def g1(mu,rho,tp,t):
  return (rho-1-mu+mu*tp)*np.exp(t)
def g2(rho,t):
  return rho*np.exp(t-1)
# integral of the density
def G1(mu,rho,tp,end):
  return max(0,(rho-1-mu+mu*tp)*(np.exp(end)-1))
def G2(rho,start,end):
  return max(0, rho*(np.exp(end-1)-np.exp(start-1)))

# first case: the CDF is first flat, then follows an exponential
def CDF_flat(mu, rho, tp, t):
  p0 = tp*(rho-1)/(1-tp)
  p1 = min(mu, 1-p0)
  tlim = 1+np.log((rho-1+p0+p1)/rho)
  if t < tlim:
    return p0
  else:
    return p0 + G2(rho, tlim, min(t,1))

# second case: the CDF follows two exponential
def CDF_2exp(mu, rho, tp, t):
  p0 = g0(mu,tp)
  p1 = min(mu, 1-p0)
  tlim = 1+np.log((rho-1+p0+p1+G1(mu,rho,tp,tp))/rho)
  if t < tlim:
    return min(1-p1, p0 + G1(mu,rho,tp, min(tp,t)))
  else:
    return min(1-p1, p0 + G1(mu,rho,tp,tp) + G2(rho, tlim, min(t,1)))

# third case: the predicted time is larger than 1
def CDF_big(mu, rho, tp, t):
  temp = g0(mu,tp) + G1(mu,rho,tp,t)
  ref = g0(mu,tp) + G1(mu,rho,tp,tp-1)
  if (ref > 1):
    return min(1, temp) # subcase where the exponential grows until a buying probability of 1
  if (t < tp-1):
    return temp # the required time belongs to the exponential growth
  if (ref > 1-mu):
    return ref # the required time is after the exponential growth, which stopped at the buying probability
  return min(temp, 1-mu) # the exponential growth stopped at 1-mu


def CDF(mu, rho, tp, t):
  if (tp >= 1):
    return CDF_big(mu, rho, tp, t)
  if (g1(mu,rho,tp,0) <= 0):
    return CDF_flat(mu, rho, tp, t)
  return CDF_2exp(mu, rho, tp, t)

# best mu in function of rho (for all prediction times)
def ParetoMu(rho):
  if(rho>1.1596):
    return (1-rho*(np.exp(1)-1)/np.exp(1))/np.log(2)
  else:
    W = lambertw(-np.sqrt((rho-1)/(4*rho)))
    return (rho-1)/(2*W)*(1+1/(2*W))


# #################### Utils for RhoMu ends here ####################
  
  
def RhoMu_paretomu(requests, pred, rho = RHO_LIMIT):
  res = [0] * len(requests)
  mu = ParetoMu(rho)
  for t, req in enumerate(requests):
    p = np.random.rand()
    if (CDF(mu,rho,pred[t]/100,1) <= p):
      res[t] = requests[t]
    else:
      buy = inverse(lambda x: CDF(mu,rho,pred[t]/100, x)) (p)
      res[t] = buy
  return res
  

# #################### Algorithms end here ####################


# #################### Combining schemes start here ####################
# Combine randomly 2 ski-rental algorithms
# algs: list of algorithms to combine
# parameterized by epsilon

def Combine_rand_discrete(requests, pred, algs, epsilon=0.1, A=1, B=B):
  diameter = 0.5
  beta = 1.0 - 0.5 * epsilon
  B_seq = B
  
  histories = list(map(lambda x: np.round(x(requests, pred)), algs))
  weights = [1] * len(algs)
  probs = [1/len(algs)] * len(algs)
  history = []
  cur_alg = random.randrange(0,len(algs))
  loss = [0] * len(algs)
  action = []
  for t, request in enumerate(requests):
    history.append(histories[cur_alg][t])
    # Compute loss of each algorithm in current step t
    loss = [B+A*x[t] if x[t] < requests[t] else A*requests[t] for x in histories]
    # loss = [B+A*x[t] / min(B,request) if x[t] < requests[t] else A*requests[t] / min(B,request) for x in histories]
    # compute the new probabilities in function of the loss and the new algorithm to follow
    new_weights = [w*(beta)**(l/diameter) for w,l in zip(weights,loss)]
    total_weights = sum(new_weights)
    new_probs = [w / total_weights for w in new_weights]
    switch_probs = compute_switch_probs_with_precision(probs, new_probs, cur_alg)
    cur_alg = np.random.choice(np.arange(0, len(algs)), p=switch_probs)
    action.append(cur_alg)
    probs = new_probs
    weights = new_weights
    weights = probs # prevents floating point errors: always normalize the weights
  return history


def Combine_rand(requests, pred, algs, epsilon=0.1):
  diameter = 0.5
  beta = 1.0 - 0.5 * epsilon
  
  istep = 0.1
  # histories = list(map(lambda x: np.round(x(requests, pred)), algs))
  histories = list(map(lambda x: x(requests, pred), algs))
  weights = [1] * len(algs)
  probs = [1/len(algs)] * len(algs)
  history = []
  cur_alg = random.randrange(0,len(algs))
  loss = [0] * len(algs)
  
  for t, request in enumerate(requests):
    previ = 0
    
    # the time is discretized in steps of size 0.1
    for i in np.arange(0,requests[t]+1,istep):
      
      # compute the new probabilities in function of the loss and the new algorithm to follow
      new_weights = [w*(beta)**(l/diameter) for w,l in zip(weights,loss)]
      total_weights = sum(new_weights)
      new_probs = [w / total_weights for w in new_weights]
      switch_probs = compute_switch_probs_with_precision(probs, new_probs, cur_alg)
      cur_alg = np.random.choice(np.arange(0, len(algs)), p=switch_probs)
      probs = new_probs
      weights = new_weights
      weights = probs # prevents floating point errors: always normalize the weights

      # the round stops before the next i
      if (histories[cur_alg][t] < i+istep or requests[t] < i+istep):
        # save the loss to update the algorithms at the next iteration
        loss = [requests[t]-previ if x[t]>=requests[t] else 0 if x[t] < previ else 1+x[t]-previ for x in histories]
        if (requests[t] < histories[cur_alg][t]):
          history.append(requests[t]+1000) # cur_alg did not buy
          break
        else:
          history.append(max (i, histories[cur_alg][t]) ) # cur_alg buys between i and the following i
          break

      loss = [i-previ if x[t]>=i else 0 if x[t] < previ else 1+x[t]-previ for x in histories]
      previ = i

  return history


def OnlineLearning(requests, pred, sigmas, s_num=10, epsilon=0.1, A=1, B=B):
  diameter = 0.5
  beta = 1.0 - 0.5 * epsilon
  beta2 = 0.8
  sigma_num = s_num
  # x_num = int(max(requests) + 1)
  x_num = 20
  scale_max = max(sigmas)
  scale_min = min(sigmas)

  history = []
  sigma_range = np.linspace(scale_min, scale_max+0.001, sigma_num)
  x_range = np.linspace(1,np.ceil(max(requests)), x_num)
  # x_range = np.arange(x_num)
  # weights of hedge algorithm
  weights = np.ones(((sigma_num-1)*(x_num),x_num)) / x_num
  
  for t, request in enumerate(requests):
    # Get index of online learning algorithm
    sigma = sigmas[t]
    pred_t = pred[t]
    s = np.digitize(np.array([sigma]), sigma_range, right=True)[0] - 1
    pt = np.digitize(np.array([pred_t]), x_range, right=True)[0] - 1
    idx = s*(x_num)+pt
    action = np.random.choice(x_range, p=weights[idx,:])
    # loss = [B+A*x if x < requests[t] else A*requests[t] for x in x_range]
    loss = [(B+A*x-1)/min(B,request) if x <= requests[t] else A*request/min(B,request) for x in x_range]
    for m in range(len(x_range)):
      weights[idx,m] = weights[idx,m] * beta2**(loss[m]/diameter)
    weights[idx,:] = weights[idx,:] / sum(weights[idx,:])

    history.append(action)
  # if sigmas[0] == 4.5:
  #   print(history[-20:])
  #   print(pred[-20:])
  return history


def RobustFTP(requests, pred):
  return Combine_rand(requests, pred, (FTP, ClassicRandom))


def RobustKumar(requests, pred):
  return Combine_rand(requests, pred, [
      FTP,
      lambda r, p : KumarRandom(r, p, 1.1),
      lambda r, p : KumarRandom(r, p, 1.1596),
      lambda r, p : KumarRandom(r, p, 1.3),
      lambda r, p : KumarRandom(r, p, 1.4),
      lambda r, p : KumarRandom(r, p, 1.5),
      ClassicRandom,])
  

def RobustRhoMu(requests, pred):
  return Combine_rand_discrete(requests, pred, [
      FTP,
      lambda r, p : RhoMu_paretomu(r, p, 1.1),
      lambda r, p : RhoMu_paretomu(r, p, 1.1596),
      lambda r, p : RhoMu_paretomu(r, p, 1.3),
      lambda r, p : RhoMu_paretomu(r, p, 1.4),
      lambda r, p : RhoMu_paretomu(r, p, 1.5),
      ClassicRandom,])


def RobustAngelo(requests, pred):
  return Combine_rand(requests, pred, [
      FTP,   
      lambda r, p : AngelopoulosDet(r, p, 1.1),
      lambda r, p : AngelopoulosDet(r, p, 1.1596),
      lambda r, p : AngelopoulosDet(r, p, 1.3),
      lambda r, p : AngelopoulosDet(r, p, 1.4),
      lambda r, p : AngelopoulosDet(r, p, 1.5),
      ClassicRandom,])
  
    
# #################### Combining schemes end here ####################


# ################### Multiple sleep states (DPM) start here ###############
# algorithms return a list of times at which they switch to the next state


def OPT_multiple(requests, pred=[]):
  history = []
  for t, req in enumerate(requests):
    history.append([])
    current_state = 0
    current_cost = requests[t]
    # go through the states and compute their cost
    for i,(power, wake) in enumerate(zip(POWER_CONSUMPTIONS,WAKE_UP_COSTS)):
      if current_cost > power*requests[t]+wake:
        current_cost = power*requests[t]+wake
        current_state = i
        
    # switch to the best state at the beginning
    for j in range(current_state):
      history[t].append(0)
  return history

# similar to OPT but based on the predictions
def FTP_multiple(requests, pred=[]):
  history = []
  for t, req in enumerate(requests):
    history.append([])
    current_state = 0
    current_predicted_cost = pred[t]
    for i,(power, wake) in enumerate(zip(POWER_CONSUMPTIONS,WAKE_UP_COSTS)):
      if current_predicted_cost > power*pred[t]+wake :
        current_state = i
        current_predicted_cost = power*pred[t]+wake
    for j in range(current_state):
      history[t].append(0)
  return history

# follows the lower envelope of the cost functions
def Online_multiple(requests, pred=[]):
  history = []
  step_power=[]
  step_wakeup = []
  for i in range(len(POWER_CONSUMPTIONS)-1):
    step_power.append(POWER_CONSUMPTIONS[i]-POWER_CONSUMPTIONS[i+1])
    step_wakeup.append(WAKE_UP_COSTS[i+1]-WAKE_UP_COSTS[i])
  for t, req in enumerate(requests):
    history.append([])
    for power,wake in zip(step_power,step_wakeup): 
      switch = wake/power
      if switch > requests[t]:
        break
      else:
        history[t].append((switch))
  return history
  
def RhoMu_multiple(requests, pred=[], rho=RHO):
  history = [] * len(requests)
  step_power=[]
  step_wakeup = []
  mu = ParetoMu(rho)
  for i in range(len(POWER_CONSUMPTIONS)-1):
    step_power.append(POWER_CONSUMPTIONS[i]-POWER_CONSUMPTIONS[i+1])
    step_wakeup.append(WAKE_UP_COSTS[i+1]-WAKE_UP_COSTS[i])
  for t, req in enumerate(requests):
    history.append([])
    p = np.random.rand()
    for power,wake in zip(step_power,step_wakeup):
      scale_pred = pred[t] * power/wake
      if (CDF(mu,rho,scale_pred,1) <= p):
        break
      else:
        scale_switch = inverse(lambda x: CDF(mu,rho,scale_pred, x)) (p)
        switch = scale_switch * wake/power
        if switch > requests[t]:
          break
        else:
          history[t].append(switch)
  return history

def RandomOnline_multiple(requests, pred=[]):
  return RhoMu_multiple(requests, [0]*len(requests), rho = np.exp(1)/(np.exp(1)-1))

def Prudent_Bp(mu, rho, pred, x):
  Bp = 0
  step_power=[]
  step_wakeup = []
  for i in range(len(POWER_CONSUMPTIONS)-1):
    step_power.append(POWER_CONSUMPTIONS[i]-POWER_CONSUMPTIONS[i+1])
    step_wakeup.append(WAKE_UP_COSTS[i+1]-WAKE_UP_COSTS[i])
  for power,wake in zip(step_power,step_wakeup):
    scale_pred = pred * power/wake
    # CDF at t' gives us value at t = (wake/power)*t'
    scale_x = x * power/wake
    Bp = Bp + wake * CDF(mu, rho, scale_pred, scale_x)
  return Bp

def RhoMu_multiple_prudent(requests, pred=[], rho=RHO):
  history = [] * len(requests)
  mu = ParetoMu(rho)
  for t, req in enumerate(requests):
    history.append([])
    for i in range(1,len(POWER_CONSUMPTIONS)):
      p = np.random.rand()
      if (Prudent_Bp(mu,rho,pred[t],0) >= WAKE_UP_COSTS[i]):
        history[t].append(0)
      elif (Prudent_Bp(mu,rho,pred[t],1) < WAKE_UP_COSTS[i]):
        break
      else:
	# we switch to i once Bp = beta[i-1] + p*(beta[i] - beta[i-1])
        switch = inverse(lambda x: Prudent_Bp(mu,rho,pred[t],x)) (WAKE_UP_COSTS[i-1] + p*(WAKE_UP_COSTS[i]-WAKE_UP_COSTS[i-1]))
        if switch > requests[t]:
          break
        else:
          history[t].append(switch)
  return history

def RandomOnline_multiple_prudent(requests, pred=[]):
  return RhoMu_multiple_prudent(requests, [0]*len(requests), rho = np.exp(1)/(np.exp(1)-1))


def Kumar_multiple(requests, pred=[], rho=RHO):
  history = [] * len(requests)
  step_power=[]
  step_wakeup = []
  for i in range(len(POWER_CONSUMPTIONS)-1):
    step_power.append(POWER_CONSUMPTIONS[i]-POWER_CONSUMPTIONS[i+1])
    step_wakeup.append(WAKE_UP_COSTS[i+1]-WAKE_UP_COSTS[i])
  lambd = inverse(lambda x : x/(1-np.exp(-x))) (rho)
  for t, req in enumerate(requests):
    history.append([])
    p = np.random.rand()
    for power,wake in zip(step_power,step_wakeup):
      scale_pred = pred[t] * power/wake
      k = lambd if scale_pred >= 1 else 1/lambd
      scale_switch = np.log(1+p*(np.expm1(k)))
      switch = scale_switch * wake/power
      if switch > requests[t]:
        break
      else:
        history[t].append(switch)
  return history

def Prudent_Bp_Kumar(lambd, pred, x):
  Bp = 0
  step_power=[]
  step_wakeup = []
  for i in range(len(POWER_CONSUMPTIONS)-1):
    step_power.append(POWER_CONSUMPTIONS[i]-POWER_CONSUMPTIONS[i+1])
    step_wakeup.append(WAKE_UP_COSTS[i+1]-WAKE_UP_COSTS[i])
  for power,wake in zip(step_power,step_wakeup): 
    scale_pred = pred * power/wake
    # CDF at t' gives us value at t = (wake/power)*t'
    scale_x = x * power/wake
    k = lambd if scale_pred >= 1 else 1/lambd
    cdf = np.expm1(scale_x) / np.expm1(k)
    if cdf > 1: cdf = 1
    Bp = Bp + wake * cdf
  return Bp

def Kumar_multiple_prudent(requests, pred=[], rho=RHO):
  history = [] * len(requests)
  lambd = inverse(lambda x : x/(1-np.exp(-x))) (rho)
  for t, req in enumerate(requests):
    history.append([])
    for i in range(1,len(POWER_CONSUMPTIONS)):
      p = np.random.rand()
      if (Prudent_Bp_Kumar(lambd,pred[t],0) >= WAKE_UP_COSTS[i]):
        history[t].append(0)
      elif (Prudent_Bp_Kumar(lambd,pred[t],1) < WAKE_UP_COSTS[i]):
        break
      else:
        switch = inverse(lambda x: Prudent_Bp_Kumar(lambd,pred[t],x)) (WAKE_UP_COSTS[i-1] + p*(WAKE_UP_COSTS[i]-WAKE_UP_COSTS[i-1]))
        if switch > requests[t]:
          break
        else:
          history[t].append(switch)
  return history


def Angelo_multiple(requests, pred=[], rho=RHO):
  history = [] * len(requests)
  step_power=[]
  step_wakeup = []
  for i in range(len(POWER_CONSUMPTIONS)-1):
    step_power.append(POWER_CONSUMPTIONS[i]-POWER_CONSUMPTIONS[i+1])
    step_wakeup.append(WAKE_UP_COSTS[i+1]-WAKE_UP_COSTS[i])
  for t, req in enumerate(requests):
    history.append([])
    for power, wake in zip(step_power, step_wakeup):
      scale_pred = pred[t] * power/wake
      scale_switch = 1 if scale_pred <= 1 else rho - 1
      switch = scale_switch * wake/power
      if switch > requests[t]:
        break
      else:
        history[t].append(switch)
  return history



# utils for Blum&Burch combination

# return the current state of the algorithm following the switches
def get_currentstate(switches, time):
  return sum([1 if x < time else 0 for x in switches])
  
# compute the cost of the algorithm following the switches from prev_time to time, count the wakeup cost difference between states
def get_residualcost(switches, prev_time, time):
  cur_state = get_currentstate(switches, prev_time)
  last_time = prev_time
  cost = 0
  for switch in switches[cur_state:]:
    if switch >= time:
      break
    cost += WAKE_UP_COSTS[cur_state+1]-WAKE_UP_COSTS[cur_state] + (switch-last_time) * POWER_CONSUMPTIONS[cur_state]
    last_time = switch
    cur_state += 1
  
  cost += (time-last_time) * POWER_CONSUMPTIONS[cur_state]
  return cost


# this is to compute EMD flow:
def compute_switch_probs(old_probs, new_probs, cur_alg):
  n = len(old_probs)

  G = nx.DiGraph()

  for i in range(n):
    G.add_node(i)
    G.add_node(n+i)

  for i in range(n):
    for j in range(n):
      if i == j:
        G.add_edge(i,n+j, capacity=1, weight=0)
      else:
        G.add_edge(i, n + j, capacity=1, weight=1)

  source = 2*n+1
  target = 2*n+2
  G.add_node(source) #source
  G.add_node(target) #target

  for i in range(n):
    G.add_edge(source,i, capacity=old_probs[i], weight=0)
    G.add_edge(n+i, target, capacity=new_probs[i], weight=0)

  flow = nx.algorithms.max_flow_min_cost(G,source, target)
  switch_probs = [ flow[cur_alg][n+i] for i in range(n) ]
  return switch_probs


def compute_switch_probs_with_precision(old_probs, new_probs, cur_alg):
  # implementation via bipartite flows sometimes fails to find the optimimum solution and breaks down.
  # In such case, I decrease the number of precision bits and try again with less precise numbers.
  precision = 53
  while precision > 0:
    N = 2**precision
    sig1 = [ round(N*p)/N for p in old_probs ]
    sig2 = [ round(N*p)/N for p in new_probs ]
    try:
      switch_probs = compute_switch_probs(sig1, sig2, cur_alg)
    except:
      precision = precision - 1
      if precision < 20:
        print("WARNING: Could not find EMD flow for\n{}\n{}, decreasing precision to {} bits".format(sig1, sig2, precision))
    else:
      break
  assert precision > 0, "Could not find EMD flow"
  total = sum(switch_probs)
  # total mass in switch_probs should be equal to the probability of staying in cur_alg state
  # this may slightly differ due to precision issues.
  # Therefore, I normalize by "total" to make sure that the probabilities sum up to 1
  switch_probs = [p / total for p in switch_probs]
  return switch_probs  


def Combine_rand_multiple(requests, pred, algs, epsilon=0.1):
  diameter = 0.5
  beta = 1.0 - 0.5 * epsilon

  istep = 0.1
  histories = list(map(lambda x: x(requests, pred), algs))
  weights = [1] * len(algs)
  probs = [1/len(algs)] * len(algs)
  history = []
  cur_alg = random.randrange(0,len(algs))
  loss = [0] * len(algs)
  uniform_metric = np.array([ [1] * len(algs)] * len(algs), dtype=np.float64)
  for i in range(0,len(algs)):
    uniform_metric[i] = 0
  
  for t, request in enumerate(requests):

    previ = 0
    history.append([])
    current_state = 0
    
    for i in np.arange(0,requests[t]+1,istep):
      new_weights = [w*(beta**(l/diameter)) for w,l in zip(weights,loss)]
      total_weights = sum(new_weights)
      new_probs = [w / total_weights for w in new_weights]
      switch_probs = compute_switch_probs_with_precision(probs, new_probs, cur_alg)
      cur_alg = np.random.choice(np.arange(0, len(algs)), p=switch_probs)

      probs = new_probs
      weights = new_weights
      weights = probs # prevents floating point errors: always normalize the weights

      nexttime = i+istep  if (requests[t] > i+istep) else requests[t]
      
      if get_currentstate(histories[cur_alg][t], nexttime) > len(history[t]):
        # follow the switches of cur_alg between i and nexttime
        for switch in histories[cur_alg][t][len(history[t]):]:
          history[t].append(max(i, switch))
      
      # stop when the request arrives
      if (requests[t] < i+istep):
        loss = [get_residualcost(x[t],previ, requests[t]) for x in histories]
        break
        
      loss = [get_residualcost(x[t], previ, i) for x in histories]
      previ = i
        
  return history



def RobustFTP_multiple_prudent(requests, pred):
  return Combine_rand_multiple(requests, pred, (FTP_multiple, RandomOnline_multiple_prudent))

def Robust_multiple(requests, pred, algorithm, rhos, robust_algorithm):
  return Combine_rand_multiple(requests, pred,
    [FTP_multiple,] + [lambda r, p : algorithm(r, p, rho) for rho in rhos] + [robust_algorithm,])

def RobustKumar_multiple(requests, pred):
  return Robust_multiple(
    requests, pred,
    Kumar_multiple,
    (1.1, 1.1595, 1.3, 1.4, 1.5),
    RandomOnline_multiple)

def RobustRhoMu_multiple(requests, pred):
  return Robust_multiple(
    requests, pred,
    RhoMu_multiple,
    (1.1, 1.1595, 1.3, 1.4, 1.5),
    RandomOnline_multiple)

def RobustAngelo_multiple(requests, pred):
  return Robust_multiple(
    requests, pred,
    Angelo_multiple,
    (1.1, 1.1595, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9),
    Online_multiple)

def RobustKumar_multiple_prudent(requests, pred):
  return Robust_multiple(
    requests, pred,
    Kumar_multiple_prudent,
    (1.1, 1.1595, 1.3, 1.4, 1.5),
    RandomOnline_multiple_prudent)

def RobustRhoMu_multiple_prudent(requests, pred):
  return Robust_multiple(
    requests, pred,
    RhoMu_multiple_prudent,
    (1.1, 1.1595, 1.3, 1.4, 1.5),
    RandomOnline_multiple_prudent)


# ################### Multiple sleep states end here ###############


# ############### Define algorithms with a prescribed rho (necessary for parallelism) ##########


def ApplyRho(alg, rho):
  algo = lambda requests, pred=[]: alg(requests, pred, rho)
#  algo.__name__ = alg.__name__ + '[rho=%0.3f]' % rho
  return algo

def RhoMu_paretomu_1_05(r,p):
  return ApplyRho(RhoMu_paretomu, 1.05)(r,p)
def RhoMu_paretomu_1_1(r,p):
  return ApplyRho(RhoMu_paretomu, 1.1)(r,p)
def RhoMu_paretomu_1_1596(r,p):
  return ApplyRho(RhoMu_paretomu, 1.1596)(r,p)
def RhoMu_paretomu_1_216(r,p):
  return ApplyRho(RhoMu_paretomu, 1.216)(r,p)
def RhoMu_paretomu_1_3(r,p):
  return ApplyRho(RhoMu_paretomu, 1.3)(r,p)
def RhoMu_paretomu_1_4(r,p):
  return ApplyRho(RhoMu_paretomu, 1.4)(r,p)
def RhoMu_paretomu_1_5(r,p):
  return ApplyRho(RhoMu_paretomu, 1.5)(r,p)
# def RhoMu_multiple_1_1596(r,p):
#   return ApplyRho(RhoMu_multiple, 1.1596)(r,p)
# def RhoMu_multiple_1_216(r,p):
#   return ApplyRho(RhoMu_multiple, 1.216)(r,p)

def KumarRandom_1_1596(r,p):
  return ApplyRho(KumarRandom, 1.1596)(r,p)
def KumarRandom_1_216(r,p):
  return ApplyRho(KumarRandom, 1.216)(r,p)
# def Kumar_multiple_1_1596(r,p):
#   return ApplyRho(Kumar_multiple, 1.1596)(r,p)
# def Kumar_multiple_1_216(r,p):
#   return ApplyRho(Kumar_multiple, 1.216)(r,p)

def AngelopoulosDet_1_1596(r,p):
  return ApplyRho(AngelopoulosDet, 1.1596)(r,p)
def AngelopoulosDet_1_216(r,p):
  return ApplyRho(AngelopoulosDet, 1.216)(r,p)


# ############## End algorithm definitions ######################

# #################### Utilities start here ####################


# compute the cost for algorithms using multiple states
def Cost_from_history(requests, history):
  res = [0]*len(requests)
  for t in range(len(res)):
    cur_state = 0
    last_switch = 0
    for switch in history[t]:
      res[t] += POWER_CONSUMPTIONS[cur_state]*(switch-last_switch)
      cur_state +=1
      last_switch = switch
    res[t] += POWER_CONSUMPTIONS[cur_state]*max(0,requests[t]-last_switch)
    res[t] += WAKE_UP_COSTS[cur_state]
  return res


# if multiple is in algname, use the multiple-state function
def Cost(history, requests, algname=""):
  assert len(history) == len(requests)
  if "multiple" in algname:
    return sum(Cost_from_history(requests,history))
  # return sum([B+x if x < req else req for x,req in zip(history,requests)])
  return sum([(B+x-1)/min(B,req) if x <= req else req/min(B,req) for x,req in zip(history,requests)])

# #################### Utilities end here ####################
