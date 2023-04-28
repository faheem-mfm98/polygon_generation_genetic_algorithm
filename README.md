# Polygon Generation Through Genetic Algorithm
Python-based genetic algorithm that creates and visualizes polygons without any intersection between its edges.

# Analysis

The program flow can be understood as being depicted in the image below:

![flowchart](https://github.com/faheem-mfm98/polygon_generation_genetic_algorithm/blob/main/images/algorithm_flowchart.png)


## runGA

runGA is a function that generates a population (list) of individual solution.

Then, it iterates over a number of functions to generate the **Best Polygon Solution**.

1. Fitness Calculation

	Calculates the fitness of all the sample polygons in the population
	based upon some criteria. 

2. Parent Selection

	Selects two parents (polygons) for the crossover to generate offspring
	polygons that are added to the population.
 
3. Crossover and Mutation

	Davies' OX1 Crossover is applied on the selected parent polygons and 
	two new offspring polygons are created and added to the population.

4. Best Solution Identification

	Fitness of the new population is calculated and the current best polygon is
	extracted and compared with the best polygon of the previous iteration.
	If success criteria is met, then the process is stopped. Else, another
	iteration occurs to generate the best polygon solution. 
	
## Individual Solution 

Each Individual solution in the population consists of:

```
chromsome: list of random 0s and 1s    -> [1,1,1]
polygon_sequence: list of 2d vertices  -> [(x1,y1),(x2,y2),(x3,y3)] 
num_intersection: num of intersections in this polygon  -> 0 
value: differnce betweem sum 1s of chromosome and num_intersection  -> 3
```

For Example, let's take a sample individual solution:

```
chromsome: [1,1,1]
polygon_sequence:  [(3,3),(10,5),(1,10)]
```
Then, creating polygon of this individual solution results in:

```
num_intersection: 0
value:  3
```

Hence, this is the best polygon solution based upon the criteria:
```
1. Polygon Value equals size of chromosome, and
2. Number of intersections equals zero.
```
