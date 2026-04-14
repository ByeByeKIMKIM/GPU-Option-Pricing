import numpy as np
import time

np.random.seed(1003)

def generate_gbm_paths(S0, mu, sigma, T, steps, paths):
    """Generate GBM paths
    Args:
        S0: initial price of the asset 
        mu: the drift constant
        sigma: the volatility
        T: time to maturity in years
        steps: number of time increments
        num_paths: the number of paths we want to simulate
    """
    delta_t = T / steps
    standard_normal_samples = np.random.standard_normal(size=(paths, steps))
    volatility_drag = (mu - (sigma ** 2) / 2) * delta_t
    inner_exp = standard_normal_samples * (sigma * np.sqrt(delta_t)) + volatility_drag
    summed = np.cumsum(inner_exp, axis=1)
    zero_start = np.concatenate((np.zeros((paths, 1)), summed), axis=1)
    paths_matrix = np.exp(zero_start) * S0
    return paths_matrix

def price_asian_call(paths, K, r, T):
    """Calculate fair price today of the asian call option
    Args:
        paths: the output of generate_gbm_paths
        K: strike price
        r: risk-free rate
        T: time to maturity in years
    """
    mean = np.mean(paths[:, 1:], axis=1)
    sub_strike = mean - K
    call_rule = np.maximum(sub_strike, 0)
    avg = np.mean(call_rule)
    discount = avg * np.exp(-r * T)
    return discount

def price_barrier_out_call(paths, K, B, r, T):
    """Calculate fair price today of down-and-out barrier call option
    Args:
        paths: the output of generate_gbm_paths
        K: strike price
        B: barrier price
        r: risk-free rate
        T: time to maturity in years
    """
    terminal_prices = paths[:, -1]
    payoffs = np.maximum(terminal_prices - K, 0)
    mask = ~np.any(paths < B, axis=1)
    survived = payoffs * mask
    avg = np.mean(survived)
    discount = avg * np.exp(-r * T)
    return discount

def main():
    S0, K, B, r, sigma, T = 100, 100, 90, 0.05, 0.2, 1.0
    steps = 252
    num_paths = 1000000

    print(f"Generating {num_paths} paths...")

    t_start = time.perf_counter()

    paths       = generate_gbm_paths(S0=S0, mu=r, sigma=sigma, T=T, steps=steps, paths=num_paths)
    asian_price   = price_asian_call(paths=paths, K=K, r=r, T=T)
    barrier_price = price_barrier_out_call(paths=paths, K=K, B=B, r=r, T=T)

    t_end = time.perf_counter()
    elapsed = t_end - t_start

    print(f"Asian Call Price:         ${asian_price:.4f}")
    print(f"Barrier Knock-Out Price:  ${barrier_price:.4f}")
    print(f"Time:                      {elapsed:.4f}s")

if __name__=="__main__":
    main()