#import matplotlib.pyplot as plotter
import numpy as np
import scipy as sc
import math

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
