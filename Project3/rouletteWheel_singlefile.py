# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:54:32 2019

@author: Tiger
"""
#from quickFinder import quickfinder
#from reproduction import haveChild
import numpy as np
import math
import random as rand
import pandas as pand

"""
The functions will be here because import does not work >:( >:(  ----------------------------------------------
"""

"""
quickfinder is the function that finds the result from a given x and y value
"""
def quickfinder(x, y):
    #Step 1
    step_1 = 1+(y*y)
    
    #Step 2
    step_2 = 10/step_1
    
    #Step 3
    toSine = math.pi*10*x+step_2
    step_3 = math.sin(toSine)
    
    #Step 4
    step_4 = (x*x)+(y*y)
    
    #Step 5
    step_5 = math.log(step_4)
    
    #Step 6
    final_answer = step_3 + step_5
    
    return final_answer

"""
formatBinary finds the binary representation of a large decimal number
    and makes sure there is the right amount of bits (14 bits)
"""
def formatBinary(decNumber):
    binNumber = format( int(decNumber),"b" )
    
    if(len(binNumber) == 13):
        binNumber = '0{}'.format(binNumber)
    if(len(binNumber) == 12):
        binNumber = '00{}'.format(binNumber)
    
    return binNumber

"""
The chances of this mutate function being performed on a string of bits
    is 1 in 100
The function takes a random index and flips the bit at that index of the string
"""
def mutate(binNumber):
    # random index for the bit to change
    ind = rand.randrange(0, len(binNumber)-1)
    
    if(binNumber[ind] == '0'):
        binNumber = binNumber[:ind]+'1'+binNumber[ind+1:]
    elif(binNumber[ind] == '1'):
        binNumber = binNumber[:ind]+'0'+binNumber[ind+1:]
    
    return binNumber

"""
haveChild does the following
- takes two x,y pairs
- finds a way to represent each pair with a 28 bit binary string
- combines the two binary strings at a random index
- breaks the new binary string into a new x,y pair
"""
def haveChild(pair1, pair2):
    # Prepare the binary representation of pair1  -----------------------------
    pair1_x = pair1['x']*1000
    pair1_y = pair1['y']*1000
    
    pair1_x_bin = formatBinary(pair1_x)
    pair1_y_bin = formatBinary(pair1_y)
    
    pair1_bin = pair1_x_bin + pair1_y_bin
    
    #               print(pair1_bin)
    
    #prepare the binary representation of pair2  ------------------------------
    pair2_x = pair2['x']*1000
    pair2_y = pair2['y']*1000
    
    
    pair2_x_bin = formatBinary(pair2_x)
    pair2_y_bin = formatBinary(pair2_y)
    
    pair2_bin = pair2_x_bin + pair2_y_bin
    
    #               print(pair2_bin)
    
    #generate a random index (call it cutoff)
    cutoff = rand.randrange(3,25)
    #the probability for a mutation to occur is 1/100
    mut_prob = rand.randrange(0,100)
    
    #create a new pair using the two binary representations -------------------
    pair3_bin = pair1_bin[:cutoff]+pair2_bin[cutoff:]
    if(mut_prob == 12):
        pair3_bin = mutate(pair3_bin)
    
    pair3_x_bin = pair3_bin[:14]
    pair3_y_bin = pair3_bin[14:]
    
    pair3_x = int(pair3_x_bin, 2)
    pair3_y = int(pair3_y_bin, 2)
    
    x_final = pair3_x/1000
    y_final = pair3_y/1000
    
    # set up the dictionay and return it
    result_final = quickfinder(x_final, y_final)
    childPair = {'x':x_final, 'y':y_final, 'result':result_final}
    
    return childPair

"""
inRange returns True if the given decimal value is in a given range
    (between low and hi)
    otherwise return False
"""
def inRange(decVal, low, hi):
    if low <= decVal <= hi:
        return True
    
    return False

"""
each time this function is called, one evolution cycle is performed on a list 
    of x,y pairs, returning a more fit list of x,y pairs
"""
def evolutionCycle(pairs):
    # the list of fitnesses for the incoming pairs ------------------------------------
    fitnesses = []
    for pr in pairs:
        fitnesses.append(pr['result'])
    
    # the list of pairs but based on probability ---------------------------------------
    selectable = []
    fit_sum = sum(fitnesses)
    
    # fill the list with the pairs
    # multiply each pair according to its probability
    for i in range(0,len(pairs)):
        p = fitnesses[i]/fit_sum
        p = int(p*1000)
        addup = [pairs[i]]*p
        selectable = selectable + addup
    
    # arrange the numbers different;y to ensure random selection
    selectable = np.random.permutation(selectable)
    
    # create a new population with new children
    nextGeneration = []
   
    while(len(nextGeneration) < len(pairs)):
        rand_index = rand.randrange(0, len(selectable))
        pair1 = selectable[rand_index]
        
        rand_index = rand.randrange(0, len(selectable))
        pair2 = selectable[rand_index]
        
        child = haveChild(pair1, pair2)
        
        # !important! only include this pair in the new population
        # if values within the range
        if( inRange(child['x'], 3.00, 10.00) and inRange(child['y'], 4.00, 8.00) ):
            nextGeneration.append(child)
    
    return nextGeneration

"""
End functions ----------------------------------------------------------------------------------------------------------
"""

from datetime import datetime
rand.seed(datetime.now())

# create an initial dataset
pairs = []
for i in range(0,101):
    this_pair = {'x':rand.randrange(3,11), 'y':rand.randrange(4,9), 'result': 0}
    this_pair['result'] = quickfinder(this_pair['x'], this_pair['y'])
    pairs.append(this_pair)

# sort the dataset, make the most fit pair appear first
s_pairs = sorted(pairs, key=lambda i: i['result'], reverse=True)

# use pandas.DataFrame to print the dataset to a file
panda_pairs = pand.DataFrame.from_dict(s_pairs, orient='columns')
panda_pairs.to_csv(r'D:\\CS457\\Wayne_Project_3\\rouletteWheel_results_small.csv', header=True, index=False, sep=',', mode='a')

# for 10 or 20 cycles, the results are sorted (fittest first) and printed to the same file
cycles = 20
for tr in range(0,cycles):
    pairs = evolutionCycle(pairs)
    s_pairs = sorted(pairs, key=lambda i: i['result'], reverse=True)
    panda_pairs = pand.DataFrame.from_dict(s_pairs, orient='columns')
    panda_pairs.to_csv(r'D:\\CS457\\Wayne_Project_3\\rouletteWheel_results_small.csv', header=True, index=False, sep=',', mode='a')
    


