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

## Us

We are five students at UC Berkeley, in a Master of Engineering program within the Fintech concentration of the IEOR department:
- [Zefan Yang](https://www.linkedin.com/in/zefan-yang-553955146/)
- [Yida Cui](https://www.linkedin.com/in/yidacui/)
- [Anqi Shi](https://www.linkedin.com/in/anqi-shi-691699180/)
- [Aakash Grover](https://www.linkedin.com/in/aakash-grover/)
- [Adam Moutonnet](https://www.linkedin.com/in/amoutonnet/)

