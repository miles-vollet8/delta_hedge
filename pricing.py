from math import sqrt, exp, log, pi
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
TRADING_DAYS = 365.0

##########################
# Black scholes calculation functions
##########################

def d1_calc(S, K, time, vol, r): #follow call equation with input parameters
    if time <= 0:
        return 0
    ln = np.log(S/K) #shows how itm or otm the option is
    numerator = (ln + (r + .5*vol*vol)*time)
    denominator = (vol*np.sqrt(time))
    d1 = numerator/denominator
    return d1
def d2_calc(d1, vol, time):
    denominator = (vol*np.sqrt(time))
    d2 = d1-denominator
    return d2

def d1(S,K,T,r,q,sig):
    return (log(S/K) + (r-q+0.5*sig*sig)*T) / (sig*sqrt(T))

def pricer(S,K,T,r,q,sig,is_call):
    if T<=0: return max(S-K,0.0) if is_call else max(K-S,0.0)
    _d1 = d1(S,K,T,r,q,sig); _d2 = _d1 - sig*sqrt(T)
    if is_call:
        return S*exp(-q*T)*norm.cdf(_d1) - K*exp(-r*T)*norm.cdf(_d2)
    else:
        return K*exp(-r*T)*norm.cdf(-_d2) - S*exp(-q*T)*norm.cdf(-_d1)

def vega_calc(d1, S, time):
    vega = S*norm.pdf(d1)*sqrt(time)
    return vega/100.0
def delta_calc(S, K, time, vol, r):
    d1 = d1_calc(S, K, time, vol, r)
    delta = norm.cdf(d1)
    return delta
def gamma_calc(S, K, time, vol, r):
    if time <= 0:
        return 0
    d1 = d1_calc(S, K, time, vol, r)
    gamma = norm.pdf(d1)/(S*vol*sqrt(time))
    return gamma
def theta_calc(S, K, time, vol, r, is_call=True):
    if time <= 0:
        return 0
    d1 = d1_calc(S, K, time, vol, r)
    d2 = d2_calc(d1, vol, time)
    common = -(S*norm.pdf(d1)*vol)/(2*sqrt(time)) 
    sub = r*K*np.exp(-r*time)*norm.cdf(d2)
    if is_call:
        theta = common - sub
    else:
        theta = common + sub
    return theta/TRADING_DAYS
def rho_calc(S, K, time, vol, r, is_call=True):
    if time <= 0:
        return 0
    d1 = d1_calc(S, K, time, vol, r)
    d2 = d2_calc(d1, vol, time)

    if is_call:
        return K * time * exp(-r * time) * norm.cdf(d2)/100.0
    else:
        return -K * time * exp(-r * time) * norm.cdf(-d2)/100.0

def vanna_calc(S, K, time, vol, r=0.0):
    if time <= 0:
        return 0
    d1 = d1_calc(S, K, time, vol, r)
    d2 = d2_calc(d1, vol, time)
    return -norm.pdf(d1) * d2 / vol

def volga_calc(S, K, time, vol, r=0.0):
    if time <= 0:
        return 0
    d1 = d1_calc(S, K, time, vol, r)
    d2 = d2_calc(d1, vol, time)
    return vega_calc(d1, S, time) * (d1 * d2) / vol

def iv_solve(target_price,S,K,time,r,q,is_call):
    if target_price<=0 or S<=0 or K<=0 or time<=0: return np.nan
    f = lambda vol: pricer(S,K,time,r,q,vol,is_call) - target_price
    try:
        return brentq(f, 1e-4, 5.0, maxiter=100)
    except ValueError as e:
        print(f"Error solving IV: {e}")
        return np.nan

def time_solver(date: str):
    today = datetime.today()
    dt = datetime.strptime(date, "%Y-%m-%d")
    time = (dt - today).days
    time = time/365
    
    return time

