from scipy.stats import norm
import numpy as np

def price_barrier_analytical(S0, K, B, r, sigma, T):
    lam = (r + sigma**2 / 2) / sigma**2
    
    # d terms for standard BS
    d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # d terms for the "reflected" path (substitute S0 → B²/S0)
    y1 = (np.log((B**2/S0) / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    y2 = y1 - sigma * np.sqrt(T)
    
    standard_european = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    reflection = S0 * (B / S0)**(2 * lam) * norm.cdf(y1) - K * np.exp(-r * T) * (B / S0)**(2 * lam - 2) * norm.cdf(y2)
    
    return standard_european - reflection

def main():
    price = price_barrier_analytical(S0=100, K=100, B=90, r=0.05, sigma=0.2, T=1.0)

    print(price)

if __name__=="__main__":
    main()