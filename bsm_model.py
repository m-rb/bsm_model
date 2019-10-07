# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:46:59 2019

@author: mbarroso
"""
import math
import numpy as np
from scipy.stats import norm

class BSM(object):
    
    def __init__(self,s0,k,T,sigma,r,q):
        self.s0 = s0
        self.k = k
        self.T = T
        self.sigma = sigma
        self.r = r
        self.q = q
        self.d1 = (math.log(s0/k) + (r-q + (sigma**2)/2)*T)
        self.d2 =  self.d1 - (sigma * np.sqrt(T))
        
    def call(self):
        price = self.s0 * np.exp(-self.q*self.T) * norm.cdf(self.d1) - (self.k * np.exp(-self.r*self.T) * norm.cdf(self.d2))
        return price
    
    def put(self):
        price = - self.s0 * np.exp(-self.q*self.T) * norm.cdf(-self.d1) +  (self.k * np.exp(-self.r*self.T) * norm.cdf(-self.d2))
        return price
    
    def quick_aprox(self):
        #moneyness should be ATM and short T; call = put
        price = 0.4 * self.s0 * self.sigma * np.sqrt(self.T)
        return price
    
    def greeks(self):
        #call
        delta = np.exp(-self.q*self.T) * norm.cdf(self.d1)
        gamma = np.exp(-self.q * self.T) * norm.pdf(self.d1) / (self.s0 * self.sigma * self.T)
        vega = self.s0 * norm.pdf(self.d1) * np.sqrt(self.T)
        return delta, gamma, vega
        
    def monte_carlo(self,number_of_sims):
        z = np.random.normal(size=number_of_sims)
        st = self.s0 * np.exp((self.r-self.q - (self.sigma**2)/2) * self.T + self.sigma * np.sqrt(self.T) * z)
        payoff = np.maximum(st-self.k,0)
        price = np.exp(-self.r*self.T) * np.mean(payoff)
        return price
    
    def quanto_adjustment(self,foreign_r,correlation,sigma_fx): 
        #usd call to euro
        #adjusted dividend yield shall be equal to : 
        #rf€- rfforeign + correlation between foreign stock and foreign currency * stdstock *stdcurrency
        quanto_adj = self.r - foreign_r + (correlation * sigma_fx * self.sigma) #Paul Wilmott Quantitative Finance Book 2nd edition pp.191
        d1 = (math.log(self.s0/self.k) + (self.r-quanto_adj + (self.sigma**2)/2)*self.T)
        d2 = d1 - (self.sigma * np.sqrt(self.T))
        price =  self.s0 * np.exp(-quanto_adj*self.T) * norm.cdf(d1) - (self.k  * np.exp(-self.r*self.T) * norm.cdf(d2))
        return price
        
    
if __name__ == "__main__":
    product = BSM(100,110,1,0.25,-0.005,0.04)
    print(product.call())
    print(product.put())
    print(product.quick_aprox())
    print(product.greeks())
    product.monte_carlo(100000)
    
    

        
        
    