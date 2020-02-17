from utils import simulation


if __name__ == "__main__":
    # Initialize the market
    market = simulation.Market()
    # Run the simulation for N steps
    N = 500
    market.simulate(N, verbose=0)
