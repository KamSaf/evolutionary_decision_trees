# Overview

Python implementation of entropy (gain ratio) based C4.5 decision trees induction algorithm in combination with genetic programming algorithm. This code was created as the practical part of computer science master's thesis.

# How to run:

1. Clone project repository:

        git clone https://github.com/KamSaf/evolutionary_decision_trees.git


2. When in project root directory, run command:

        python3 main.py


# Genetic Algorithm (Genetic Programming) hiperparameters:

| HIPERPARAMETER  |  VALUE  | DEFAULT VALUE | DESCRIPTION |
| ------------- |  -------------  | ------------- | ------------- |
| ```dataset``` |  ```Dict[int, Dict[str, str | float]]```  | N/A | Dataset as a dictionary |  
| ```fitness_metric``` |  ```"accuracy"``` or ```"f1_score"```  | ```"accuracy``` | Metric which is going to be used in fitness calculation |
| ```population_size``` |  ```int```  | ```100``` | Number of decision trees in population |  
| ```generations``` |  ```int```  | ```50```  | Number of genetic algorithm iterations |
| ```crossover_rate``` |  ```float```  | ```0.6``` | Probability of crossover taking place between selected parents |
| ```mutation_rate``` |  ```float```  | ```0.1``` | Probability of mutation taking place in offspring |
| ```selection_method``` |  ```"tournament"``` or ```"roulette"```  | ```tournament``` | Method used in parents selection in reproduction process |
| ```tournament_size``` |  ```int```  | ```5``` | Number of trees selected to compete in tournament |
| ```max_tree_depth``` |  ```int```  | ```6``` | Maximum depth of each decision tree in population |
| ```random_pop_ratio``` |  ```float```  | ```0.75``` | Proportion of randomly generated decision trees in population |
| ```train_ds_ratio``` |  ```float```  | ```0.7``` | Proportion of dataset to include in training set |
| ```valid_ds_ratio``` |  ```float```  | ```0.15``` | Proportion of dataset to include in validation set |
| ```test_ds_ratio``` |  ```float```  | ```0.15``` | Proportion of dataset to include in testing set |
| ```elite_num``` |  ```int```  | ```5``` | Number of best decision trees in population to be transfered without changes to new generation  |


# Technologies and tools used:

- Python 3.10.12,

# Technologies and tools used (development):

- Flake8,
- Pylance
