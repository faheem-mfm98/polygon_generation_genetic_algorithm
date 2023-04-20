#!/usr/bin/env python
# coding: utf-8

# In[8]:


################################ Libraries #################################################
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import collections
################################# Variables & Constants #################################################

population_size = 0             # no of individuals
max_generations = 0             # no of generations
number_of_points = 0            # size of chromosome or no of vertices or sides of a polygon
crossover_probability = 0.0
mutation_probability = 0.0
set_of_points = []              # list of vertices or 2d_points

################################# Logic ####################################################

class Edge:
    def __init__(self,A,B):    # Edge between point A (ax, ay) and B (bx, by)
        self.point_A = A
        self.point_B = B
        
        
    def slope(self):           # slope of the Edge
        denominator = self.A[0] - self.B[0]
    
        if denominator == 0:
            return None
    
        return (self.A[1]-self.B[1])/denominator
    
    def get_line_coefficients(self):         # line: Ax + By = C
            
        A = self.point_B[0][1]-self.point_A[0][1]          # y2-y1
        B = self.point_A[0][0]-self.point_B[0][0]          # x1-x2
        C = A*self.point_A[0][0] + B*self.point_A[0][1]    # Ax1+By1
    
        return A,B,C

    
def has_common_point(edge_1,edge_2):
    
    for point in [edge_1.point_A[0],edge_1.point_B[0]]:
        if point in [edge_2.point_A[0],edge_2.point_B[0]]:
            return True
    
    return False

def is_on_edge(point, edge):
    
    # if min(x1,x2) <= x <= max(x1,x2) and min(y1,y2) <= y <= max(y1,y2)
    # then the point is on the edge
    if min(edge.point_A[0][0],edge.point_B[0][0]) <= point[0] <= max(edge.point_A[0][0],edge.point_B[0][0]):
        if min(edge.point_A[0][1],edge.point_B[0][1]) <= point[1] <= max(edge.point_A[0][1],edge.point_B[0][1]):
            return True
    return False
                
def check_intersection(edge_1,edge_2):  # check intersection btw 2 line segments or edges
    
    if(has_common_point(edge_1,edge_2)):
        return False                               # no intersection between edges having common vertex point
    
    A1,B1,C1 = edge_1.get_line_coefficients()      # get the coefficients of line equation
    A2,B2,C2 = edge_2.get_line_coefficients()      # both the edges

    det = (A1 * B2) - (A2 * B1)           

    if (det == 0):          # if lines are parallel
        return False        # edges do not intersect
    else:
        # find the point (x,y) at which the two edges intersect
        x = ((B2 * C1) - (B1 * C2)) / det
        y = ((A1 * C2) - (A2 * C1)) / det

        # if the point is a common vertex point to both edges,
        # then no intersection takes place
        if (x,y) in [edge_1.point_A[0],edge_1.point_B[0]]:
            if (x,y) in [edge_2.point_A[0],edge_2.point_B[0]]:
                return False

        # else if the point is on both the the edges,
        # then the edges intersect
        return is_on_edge((x,y), edge_1) and is_on_edge((x,y),edge_2) 

def draw_polygon(points,chromosome):
    xx = []
    yy = []
    count = 0
    first_index = -1
    flag = False
    
    for i in range(0,len(points)):
        if chromosome[i] == 1:
            xx.append(points[i][0][0])
            yy.append(points[i][0][1])
            count+=1
            if flag == False:
                first_index = i
                flag = True
            
    
    xx.append(points[count-1][0][0])
    yy.append(points[count-1][0][1])
    if first_index > -1:
        xx.append(points[first_index][0][0])
        yy.append(points[first_index][0][1])
    plt.plot(xx, yy)
    plt.show()


def euc_dist(point1,point2):
    return pow(pow(point2[0][0]-point1[0][0],2)+pow(point2[0][1]-point1[0][1],2),0.5)

def sort_insert(set_of_points,point):
    index = 0 
    length = len(set_of_points)
    
    if length == 0:
        set_of_points.append(point)
        return 
    
    while index < length:  
        if point[1] < set_of_points[index][1]:
            set_of_points.insert(index,point)
            return
        index = index + 1
    
    set_of_points.append(point)

    return

#################################### GA Section #####################################

Individual = collections.namedtuple('population', 'chromosome polygon_sequence num_intersections value')

# Generate Population , given the size and backback_Capacity

def rand_seq(set_of_points,no_of_points):
    seq = []
    index_flags = [False]*no_of_points
    i = 0
    while i < no_of_points:
        index = random.randint(0,no_of_points)
        
        if index_flags[index] == False:
            seq.append(set_of_points[index])
            index_flags[index] = True
            i += 1
    return seq

def generate_population(size, no_of_points):
    new_population = []

    for i in range(size):    
        ### Initialize Random population
        new_population.append(
            Individual(
                chromosome=random.randint(2, size=(1, no_of_points))[0],
                polygon_sequence=rand_seq(set_of_points,no_of_points),
                num_intersections=0,
                value=0
            )
        )                        

    return new_population

def create_polygon(individual):
    
    if sum(individual.chromosome) <= 3:     # polygon require at least 3 points
        return 0
    
    chrom_length = len(individual.chromosome)
    edges = []
    count_intersections = 0
    first_index = -1
    index_1 = 0
    index_2 = 1
                    
    flag = False
    while index_1 < chrom_length and index_2 < chrom_length:
        if index_2%chrom_length == first_index:
            if first_index != index_1:
                if individual.chromosome[first_index] == 1 and individual.chromosome[index_2%chrom_length] == 1:
                    new_edge = Edge(individual.polygon_sequence[index_1],individual.polygon_sequence[first_index])
                    for edge in edges:
                        if check_intersection(new_edge,edge) == True:
                            count_intersections += 1
            break
            
        if individual.chromosome[index_1] == 0:
            index_1 += 1
            index_2 += 1
        else:
            if flag == False:
                first_index = index_1
                flag = True

        if index_2 >= chrom_length:
            index_2 = 0
            continue
            
        if individual.chromosome[index_2] == 0:
            index_2 += 1
            continue
            
        new_edge = Edge(individual.polygon_sequence[index_1],individual.polygon_sequence[index_2])
        for edge in edges:
            if check_intersection(new_edge,edge) == True:
                count_intersections += 1
                            
        edges.append(new_edge)
        
        index_1 = index_2
        index_2 += 1
        if index_2 >= chrom_length:
            index_2 = 0

    return count_intersections
                     
def get_value(chromosome):
    return sum(chromosome)
    
    
def calculate_fitness(population):
    value = 0
    num_intersection = 0
    for individual in range(len(population)):
        num_intersection = 0
        num_intersection = create_polygon(population[individual])
        chrom_value = sum(population[individual].chromosome)
      
        population[individual] = Individual(chromosome=population[individual].chromosome,
                                            polygon_sequence=population[individual].polygon_sequence,
                                            num_intersections=num_intersection,
                                            value=chrom_value)        
    return population
            

def apply_mutation(chromosome,size,mutation_probability):
    
    if random.randint(0, 100) <= mutation_probability * 100:
        genes = random.randint(0, 2)

        for i in range(genes):
            gene = random.randint(0, size-1)
            if chromosome[gene] == 0:
                chromosome[gene] = 1

    return chromosome

def apply_crossover(population):          # Davis' Order Crossover (OX1), permutation based crossover
    crossovered_population = []
    num_points = len(set_of_points)

    while len(crossovered_population) < len(population):
        if random.randint(0, 100) <= crossover_probability * 100:
            parent_a = random.randint(0, len(population) - 1)
            parent_b = random.randint(0, len(population) - 1)
            
            point_1 = random.randint(0,num_points/2)         ### cut off point1 
            point_2 = random.randint(num_points/2,num_points)### cut off point2

            ##### chromosome and polygon sequence for two child a and b
            chromosome_a = [None]*num_points
            chromosome_b = [None]*num_points
            polygon_sequence_a = [None]*num_points
            polygon_sequence_b = [None]*num_points

            for i in range(point_1,point_2):      #  Davis' OX1 Crossover
                chromosome_a[i] = population[parent_a].chromosome[i]
                chromosome_b[i] = population[parent_b].chromosome[i]
                polygon_sequence_a[i] = population[parent_a].polygon_sequence[i]
                polygon_sequence_b[i] = population[parent_b].polygon_sequence[i]

            j = point_2        
            k = point_2
            for i in range(point_2,point_2+num_points):
                index = i%num_points

                if population[parent_b].polygon_sequence[index] not in polygon_sequence_a:
                    
                    polygon_sequence_a[j%num_points] = population[parent_b].polygon_sequence[index]
                    chromosome_a[j%num_points] = population[parent_b].chromosome[index]
                    j += 1

                if population[parent_a].polygon_sequence[index] not in polygon_sequence_b:    
                    
                    polygon_sequence_b[k%num_points] = population[parent_a].polygon_sequence[index]
                    chromosome_b[k%num_points] = population[parent_a].chromosome[index]
                    k += 1
                    
            chromosome_a = apply_mutation(chromosome_a, num_points, mutation_probability)
            chromosome_b = apply_mutation(chromosome_b, num_points, mutation_probability)
            #### offspring_a
            ind_a = Individual(
                chromosome=chromosome_a,
                polygon_sequence = polygon_sequence_a,
                num_intersections=0,
                value=0
            )
            
            chromosome_a_intersections = create_polygon2(ind_a)
            chrom_val = sum(chromosome_a)-chromosome_a_intersections
            crossovered_population.append(Individual(
                chromosome=chromosome_a,
                polygon_sequence = polygon_sequence_a,
                num_intersections=chromosome_a_intersections,
                value=chrom_val
            ))
            ##### offspring_b
            ind_b = Individual(
                chromosome=chromosome_b,
                polygon_sequence = polygon_sequence_b,
                num_intersections=0,
                value=0
            )

            chromosome_b_intersections = create_polygon2(ind_b)
            chrom_val = sum(chromosome_b)-chromosome_b_intersections
            crossovered_population.append(Individual(
                chromosome=chromosome_b,
                polygon_sequence = polygon_sequence_b,
                num_intersections=chromosome_b_intersections,
                value=chrom_val
            ))

    return roulette_wheel(population + crossovered_population,len(population)*2)
    
def roulette_wheel(crossovered_population,size):
    total_value = 0
    new_pop = []
    
    for individual in crossovered_population:
        total_value += individual.value
        
    probs = []
    for ind in range (size):
        probs.append((ind,round(crossovered_population[ind].value/total_value,5)))
    
    while len(new_pop) < size/2:
        i = random.randint(0,size)   # random index means wheel rotating
        
        if probs[i][1] >= random.random() * 100:   # at which section the wheel stops 
            new_pop.append(crossovered_population[probs[i][0]])
            
    return new_pop

def parent_selection(population):
    parents = []
    total_value = 0

    for individual in population:
        total_value += individual.value

    # Find Fittest Individual to select parent
    highest, second_highest = find_two_fittest_individuals(population)
    
    parents.append(highest)
    parents.append(second_highest)

    ### Check Total sum value of fittest individuals
    sum_value = 0
    while len(parents) < len(population):
        individual = random.randint(0, len(population)-1)
        sum_value += population[individual].value

        if sum_value >= total_value:
            parents.append(population[individual])

    return parents
 
def find_two_fittest_individuals(population):
    pop_size = len(population)
    
    highest_index = 0
    second_highest_index = 0
    
    for i in range(1,pop_size):
        if population[highest_index].value < population[i].value:
            second_highest_index = highest_index
            highest_index = i
            
        elif highest_index == second_highest_index:
            second_highest_index = i
            
        elif population[second_highest_index].value < population[i].value:
            second_highest_index = i
    
    return population[highest_index], population[second_highest_index]


def runGA():
    population = generate_population(population_size,len(set_of_points))

    value = []
    iteraction = []
    best_solution = None
    print("GA")
    
    for i in range(max_generations):       
        print("\nGeneration: ",i+1)
        ## Calculate Fitness of initial population
        fitness = calculate_fitness(population)
        print("###### Fitness Applied #######\n")
        ## Select parent
        parents = parent_selection(fitness)
        print("###### Parent Selection Applied ########\n")
        ## Apply crossover and mutation
        crossovered = apply_crossover(parents)
        print("##### Crossover & Mutation Applied ########\n")
        ## Calculate Fitness of population
        population = calculate_fitness(crossovered)
        #roulette-wheel
        
        ## Find fittest cadidates                
        candidate, _ = find_two_fittest_individuals(population)
        print("Generation ",i+1," Fittest:")
        print(candidate)
        
        draw_polygon(candidate.polygon_sequence,candidate.chromosome)
        
        if best_solution is None:
            best_solution = candidate
        elif candidate.value - candidate.num_intersections > best_solution.value - best_solution.num_intersections :
            best_solution = candidate

        value.append(best_solution.value)
        iteraction.append(i)

        print("Best Solution So Far\n")
        print("Chromosome: ",best_solution.chromosome)
        print("polygon sequence: ",best_solution.polygon_sequence)
        print("Value (num_1s - num_intersections): ",best_solution.value,"-",
              best_solution.num_intersections,"= ",
              best_solution.value-best_solution.num_intersections)
        
        if best_solution.value == len(set_of_points) and best_solution.num_intersections == 0:
            break

    if best_solution != None:
        print("Solution!!!\n")
        print("Chromosome: ",best_solution.chromosome)
        print("polygon sequence: ",best_solution.polygon_sequence)
        print("Value (num_1s - num_intersections): ",best_solution.value-best_solution.num_intersections)
    else:
        print("No solution")
    return

    
def main():

    ############################ Variables Initialization ####################
    global population_size 
    population_size = random.randint(50, 100)
    global max_generations 
    max_generations = random.randint(100, 500)
    
    global crossover_probability 
    crossover_probability = round(random.uniform(low=0.3, high=1.0), 1)
    global mutation_probability 
    mutation_probability = round(random.uniform(low=0.0, high=0.5), 1)
    
    number_of_points = random.randint(3,16)    # no. of points: 3-15   
    global set_of_points                         # list of points

    ############################ Set of Points (Domain) Initialization ####################
    i = 0
    while i < number_of_points:
        point = [(random.randint(0,51),random.randint(0,51)),0]   # random polygon point (vertex) 
        position_vector = euc_dist([(0,0),0],point)   # position vector of the current point
        point[1] = position_vector

        if point not in set_of_points:
            sort_insert(set_of_points,point)          # sorted insertion 
            i += 1
            
    ########################### Genetic Algorithm Called ##################################
    print("Polygon Domain Points")
    print(set_of_points)
    chromosome = random.randint(2,size=(1,number_of_points))[0]   # chromosome selects the points from the Domains
    
    
    print('\n\n--- Generated Parameters -----')
    print('Population size......: {}'.format(population_size))
    print('Number of generations: {}'.format(max_generations))
    print('Crossover probability: {}'.format(crossover_probability))
    print('Mutation probability.: {}'.format(mutation_probability))
    print('Number of Points.....: {}'.format(number_of_points))
    print('Set of Points........: {}'.format(set_of_points))
    runGA()
    
    
if __name__ == "__main__":
    main()


# In[ ]:




