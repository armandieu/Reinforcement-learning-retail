# Retail Restock Reinforcement Learning Project

## Overview
This project implements a reinforcement learning solution for optimizing retail inventory management and restocking decisions. The system uses deep learning techniques to develop intelligent restocking policies that balance inventory costs, storage constraints, and customer demand.

## Project Structure



## Quick Start
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebook: `jupyter notebook retail_restock_project.ipynb`



# Environment Configuration

## Overview
The simulation environment is configured to mimic a typical grocery shop setting, with parameters such as initial stock capacity, maintenance costs, expiration times, buying and selling prices, weekly demand, and maximum time horizon specified.

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| Capacity | Maximum stock the shop can hold |
| Buying Price | Price at which products are purchased |
| Selling Price | Price at which products are sold |
| Weekly Demand | Expected demand for products per week |
| Expiration Time | Shelf life of products in days |
| Max Time | Maximum time horizon for the simulation |
| Gamma | Discount factor for future rewards |
| Expiration Cost | Cost associated with expired products |
| Max Loss | Maximum acceptable loss for the shop |
| Maintenance Cost | Cost associated with maintaining the inventory |

## Implementation Example
```python
class RetailEnvironment:
    def __init__(self,
                 capacity=100,
                 buying_price=1.0,
                 selling_price=2.0,
                 weekly_demand=50,
                 expiration_time=7,
                 max_time=365,
                 gamma=0.95,
                 expiration_cost=1.5,
                 max_loss=1000,
                 maintenance_cost=0.1):
        
        self.capacity = capacity
        self.buying_price = buying_price
        self.selling_price = selling_price
        self.weekly_demand = weekly_demand
        self.expiration_time = expiration_time
        self.max_time = max_time
        self.gamma = gamma
        self.expiration_cost = expiration_cost
        self.max_loss = max_loss
        self.maintenance_cost = maintenance_cost
```
## Retail Environment
The retail environment simulates a real-world inventory management system with the following components:

## State Space
The state space in this retail environment represents the current status of the inventory system through a tuple structure.

### State Components
```python
state = (current_stock, time_to_expiration)
```

### State Space
- Discrete state space
- Finite number of possible states
- Size = (max_capacity + 1) × (max_expiration_time + 1)

### Action Space
The action space consists of possible restocking quantities at each time step.
- Discrete restock quantities
- Range: 0, max_stock_capacity
- Type: Integer
- Represents quantity to restock

### Reward Structure
```python
Total reward = Utility - Waste Penalty - Maintenance Cost - Unavailability penalty

Utility = (SP - C) * d
Maintenance_Cost = MC * US
Waste_Penalty = WC * UW
Unavailability_Penalty = SP * DNC
DNC = max(AS + Action - d, 0)
```
# Variable Definitions

| Variable | Description | Symbol |
|----------|-------------|---------|
| Demand | Poisson distributed demand | d |
| Actual Stock | Current inventory level | AS |
| Selling Price | Unit selling price | SP |
| Cost | Unit cost of product | C |
| Waste Cost | Cost per wasted unit | WC |
| Units Wasted | Number of units wasted | UW |
| Maintenance Cost | Cost per unit for storage | MC |
| Units in Stock | Current inventory level | US |
| Demand Not Covered | Unmet demand | DNC |
