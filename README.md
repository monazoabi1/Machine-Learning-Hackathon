# Machine Learning Hackathon
This project focuses on improving public transportation in Israel using machine learning. 

## HU.BER transportation
HU.BER transportation improvements: Provide insights and practical solutions to improve the public transportation system using machine learning techniques 
The dataset includes over 226,000 bus stops, capturing various features like passenger boarding and trip durations. 
The tasks involve predicting the number of passengers boarding at specific bus stops and estimating the total trip duration for buses. Using these predictions, the project aims to provide practical insights and suggestions to optimize the transportation system, such as adjusting bus frequencies or proposing new routes. 

## Code Files:
### main_subtask1: Predicting Passenger Boardings at Bus Stops**

The goal of this task is to predict the number of passengers boarding a bus at a given stop. 

**Input:**  
A CSV file where each row contains information about a specific bus stop along a route, excluding the column `passengers_up` which represents the number of passengers boarding.

**Output:**  
A CSV file named `passengers_up_predictions.csv`, containing two columns:  
1. `trip_id_unique_station`  
2. `passengers_up`

This output will provide the predicted number of passengers boarding for each bus stop.

### main sub_task2: Predicting Trip Duration

In this task, the goal is to predict the total duration of a bus trip, from its first station to the last. Each bus trip, identified by a unique `trip_unique_id`, is treated as a single sample. Based on the information from all bus stops within the trip, the objective is to predict the arrival time at the final stop.

**Input:**  
A CSV file where each row represents a single bus stop within a specific trip. The test set excludes the arrival times at the stops, except for the first station, which provides the departure time.

**Output:**  
A CSV file named `trip_duration_predictions.csv`, containing two columns: 
- `trip_id_unique` 
- `trip_duration_in_minutes` 

This output will predict the total trip duration in minutes.ip_duration_example.csv`.
