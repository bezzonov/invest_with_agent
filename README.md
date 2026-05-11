# 📊 Adaptive Trading Strategies with Reinforcement Learning (Russian Stock Market)
This project presents a comprehensive approach to data collection and analysis for developing adaptive trading strategies in the Russian stock market.

## 📥 Data Sources
To ensure data completeness and relevance, two primary sources were used:

- Tinkoff API — for historical trading data on major Russian stocks

- Moscow Exchange ISS (Information & Statistical Server) — for stock index data

## 🤖 Models & Methods
The core of the research focuses on modern deep learning architectures and reinforcement learning (RL) algorithms. 

Implemented agents include:

- Advantage Actor-Critic (A2C)

- Proximal Policy Optimization (PPO)

- Deep Deterministic Policy Gradient (DDPG)

- Soft Actor-Critic (SAC)

= Twin Delayed DDPG (TD3)

## 🖥️ Interactive Interface
A user-friendly web application was built using Streamlit, providing an intuitive interface for interacting with the models.

## Key features:

- Select a custom list of stocks

- Set initial capital

- Define trading period

- Choose RL algorithm for training

- Run training and strategy generation with a single click

## 📈 Evaluation
The developed strategies were evaluated against:

- Mean-Variance Optimization (MVO) — classical portfolio optimization method

- Market indices — for relative performance benchmarking

## 💡 Results & Insights
The analysis showed that:

- Certain stocks and sectors consistently demonstrate higher profitability and activity

- RL-based strategies can outperform traditional approaches under real market conditions

### These insights can be useful for:

- Individual investors

- Quant researchers

- Portfolio managers seeking to integrate ML-driven strategies

## 🚀 Tech Stack
- Python
- Pandas, Numpy, Scipy
- PyTorch / RL frameworks (e.g., Stable Baselines3)
- Streamlit
- Tinkoff API
- MOEX ISS
<img width="603" height="468" alt="image" src="https://github.com/user-attachments/assets/8f13eaf8-a06c-4983-9697-4b7c147eb275" />

<img width="683" height="409" alt="image" src="https://github.com/user-attachments/assets/ab77a055-1c55-4d3c-8ca8-2a96219f7d16" />

