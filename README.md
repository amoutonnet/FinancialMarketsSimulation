# Simulating Financial Markets Using Reinforcement Learning

## Motivation

Currently, algorithm trading has gained immense popularity in recent years. In addition, machine learning is also gaining increasing importance in the financial market, where people usually use it to predict price movements of the asset and make an optimal trading strategy. However, a number of limitations of traditional machine learning have been exposed. For instance, the financial market is very unpredictable as it constantly changes based on the events. As a result, methods that rely on historical data tend to have low prediction accuracy. Moreover, it is also difficult for classical machine learning algorithms to determine the thresholds for optimal policy and convert the predictions into actions. Therefore, our group is trying to leverage a different machine learning paradigm, one of multi-agent reinforcement learning, where the agents themselves learn the improved techniques by accounting for the changes in the environment.

## Plan

Inspired by the idea of [Alpha Zero](https://arxiv.org/abs/1712.01815), we aim to leverage the idea that the self-taught agents could become much more powerful than agents trained in a supervised manner. Our proposal could be divided into two phases:

### Phase 1

We create a ‘game’ with a set of rules to simulate financial markets (e.g. an index of 50 stocks). The game would comprise of two types of agents:

- Market makers will fix a bid and an ask price with the volume they are ready to buy or sell at each timestep. They will try to maximize their profit earned with the spread by buying stocks at the bid price to dealers, and selling it at the ask price to other dealers. A great challenge for the market makers will be to respect the supply and demand law in order for a market to exist.
- Dealers will trade stocks on the market to maximize their profit, they will have to abide by market makers prices to interact with the market. At each time step, they will be able to buy or sell stocks at the market price, or to fix a limit above (resp below) which they will sell (resp buy) stocks.

Our focus for this phase would be to ensure the stability of the environment and its ability to scale up. We aim to use the research papers in ‘fairness’ and ‘mean-field game theory’ to achieve this objective.

### Phase 2

To resemble a real environment, we would add externalities to the current system to create a better simulation of the real-world world. Furthermore, we will create a platform (website) where people will be able to try their hand at trading against the bots.

## The Simulation

Here are the main points of the simulation. There is an initialization phase, and then time steps after time steps agents will interact together.  
  
**Note**: A selling order is an order stating that a dealer wants to sell a stock at the market price. A buying order is an order stating that a dealer wants to buy a stock at the market price.

### Initialization

- We create a given number ***n*** of companies, each company is assigned a market maker that will fix the market price for this and only this company
- Each company issue a given number ***m*** of stocks
- We create a given number ***n<sub>d</sub>*** of dealers
- Each dealer is given ***m/n<sub>d</sub>*** stocks of each company (***m*** is a multiple of ***n<sub>d</sub>***)
- Each dealer is given an amount ***c*** of cash
- We initialize the price of each stock on the market to ***p***

### Market Makers Global Description

Here are a few assumptions about market makers.

- At each time step they will fix an ask price for the company they are making the market for and release the number of stocks they have available to sell at the beginning of the time step (which can evolve during it, as selling orders are processed first).
- The spread (difference between the ask price and the bid price) is a constant equal to ***s*** dollars. It is also representative of the trading cost for the dealers. We decided to make is constant to simplify things.
- They have a portfolio in which they can store temporarly the stocks of their own company they bought and have not sell yet between time steps.
- They have infinite money ressources to buy stocks to dealers and they are always ready to buy stocks no matter the quantity. Therefore, for every buying order passed by dealers the transaction is sure to happen.
- Market makers only care about their and only their company. The share price of one company does not affect the share price of others for the moment. Maybe we can implement later a way to link companies together depending on their "similarities".

### Dealers Global Description

Here are a few assumptions about dealers.

- At each time step they decide first whether they want to trade or do nothing. In the first case, they decide which company they want to trade, then which action they want to take (buy or sell at the market price) and then the amount of the transaction (how many stocks they want to trade).
- Their cash amount cannot go negative.
- They cannot sell stock they do not have (for the moment we do not implement a shorting action, maybe in phase 2).
- They can trade only one company at a time, maybe we can implement later more complex actions where they can trade multiple company during a single time step.

### Time Steps

Here is the progress of each time step:

1. Market makers make the market.
2. Dealers take actions. Each trading decision creates an order summarizing the details of the incoming transaction.
3. Every order is processed (selling orders first, then buying orders) to the extend of stock and cash availability for dealers. Indeed, dealers can decide to sell stocks they do not have, buy without cash or buy stocks that are not available, but such action will be penalized and the transaction will not happend.
4. Go back to step 1.

### Market Makers Observations

Here are a few assumptions about what market makers based their decisions on during a given time step (the observation of the environment they have access to). Within a given temporal window of length ***T<sub>w</sub>*** finishing at the last time step, they will have access to:

- The state of the market (ask price) of their own company.
- The movements on the market (sales and purchases) of their own company.
- The state of their portfolio.

### Dealers Observations

Here are a few assumptions about what dealers based their decisions on during a given time step. Within a given temporal window of length ***T<sub>w</sub>*** finishing at the last time step, they will have access to:

- The state of the market (ask price) for each company.
- The released number of stocks market makers had available to sell (for each company).
- The state of their portfolio (how many stocks owned, for which company).
- The owned amount of cash.

Plus, as market makers make the market at the beginning of the time step before dealers take their action, they will also have access to:

- The current state of the market
- The current released number of stocks market makers have available to sell (for each company).

## Us

We are five student at UC Berkeley, in a Master of Engineering program within the Fintech concentration of the IEOR department:

- [Adam Moutonnet](https://www.linkedin.com/in/amoutonnet/)
- [Zefan Yang](https://www.linkedin.com/in/zefan-yang-553955146/)
- [Yida Cui](https://www.linkedin.com/in/yidacui/)
- [Anqi Shi](https://www.linkedin.com/in/anqi-shi-691699180/)
- [Aakash Grover](https://www.linkedin.com/in/aakash-grover/)
<<<<<<< HEAD
=======
- [Adam Moutonnet](https://www.linkedin.com/in/amoutonnet/)
>>>>>>> a3b3d7c91dc39ffbaf1fdc029ba91ffe2c22de26
