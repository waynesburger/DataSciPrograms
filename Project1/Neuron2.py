#import matplotlib.pyplot as plotter
import numpy as np
import csv

class InputVect(object):
    elements = []
    expectLet = None
    expectVal = None
    
    def __init__(self, els, expL, expV):
        self.elements = els
        self.expectLet = expL
        self.expectVal = expV

class Neuron(object):
    
    weights = []
    thresh = 0
    lrn_rate = 0
    
    def __init__(self, strt_weights, strt_thresh, strt_lrn):
        #self.weights = strt_weights.transpose()
        self.weights = strt_weights
        self.thresh = strt_thresh
        self.lrn_rate = strt_lrn
        print("--------------------------------")
        print("\t\tmy weights are {}".format(self.weights.transpose() ) )
        print("\t\tmy threshold is {}".format(self.thresh))
        print("\t\tmy learing rate is {}".format(self.lrn_rate))
        print("--------------------------------")
    
    def analyze_input(self, in_vector):
        trans_weights = self.weights.transpose()
        
        result_mtrx = np.matmul(trans_weights, in_vector)
        sumup = result_mtrx[0][0]
        
        print("--------------------------------")
        print("\t\tThe vector I got: {}".format(in_vector.transpose() ) )
        print("\t\tThe sum I came up with: {}".format(sumup) )

        if(sumup > self.thresh):
            print("\t\tI'll spit out a 1 ...")
            print("--------------------------------")
            return 1
        else:
            print("\t\tI'll spit out a 0 ...")
            print("--------------------------------")
            return 0

    def modify(self, in_vector, diff):
        err = self.lrn_rate*diff
        temp_vector = np.multiply(err, in_vector)
        self.weights = np.add(self.weights, temp_vector)
        print("--------------------------------")
        print("\t\tI modified my weights, theyre now: {}".format(self.weights.transpose() ) )
        print("--------------------------------")

#--------------------------------------------------------------------------------------------------------------------

class NeuroActivity(object):
    inputs = []
    nero = None

    def __init__(self, ins, n):
        self.inputs = ins
        self.nero = n

    def single_round(self):
        pass

class NeuroTraining(NeuroActivity):
    
    def __init__(self, ins, n):
        self.total = len(ins)
        self.tot_correct = 0
        NeuroActivity.__init__(self, ins, n)

    def single_round(self):
        for single_input in self.inputs:
            print("\n=================================================")
            print("Putting {} into nero... expecting {}".format(single_input.elements.transpose(), single_input.expectVal))
            
            output = self.nero.analyze_input(single_input.elements)
            difference = single_input.expectVal - output
        
            print("output is ... {}".format(output))
            if difference != 0:
                print("modifying nero's weights ... ...")
                self.nero.modify(single_input.elements, difference)
            
            print("=================================================\n")
            
    def get_trained_neuron(self):
        return self.nero

class NeuroTesting(NeuroActivity):

    def __init__(self, ins, n):
        NeuroActivity.__init__(self, ins, n)

    def single_round(self):
        for single_input in self.inputs:
            print("\n=================================================")
            print("Putting {} into nero... expecting {}".format(single_input.elements.transpose(), single_input.expectVal))

            output = self.nero.analyze_input(single_input.elements)

            print("output is ... {}".format(output))
            if(output != single_input.expectVal):
                print("Incorrect! a {} was expected".format(single_input.expectLet) )
            else:
                print("Correct! a {} was expected".format(single_input.expectLet) )
            print("=================================================\n")

#-------------------------------------------------------------------------------------------------------------------

def training_func(fileName, strt_weights):

    # This is Nero, say hi nero
    nero = Neuron( np.array(strt_weights) , 0, 0.3)

    #------------------ Setup -------------------------------
    
    # the array of the training inputs
    train_ins = []
    # the array of the test inputs
    test_ins = []
    
    # Read the CSV file here
    with open(fileName) as open_file:
        fileReader = csv.reader(open_file, delimiter=',')
        currentLine = 0
        lineCount = 0
        arrSize = 0

        for lin in fileReader:
            # The very first line contains the number of arrays to make
            # as well as the size of the arrays
            if(currentLine == 0):
                lineCount = int(lin[0])
                arrSize = int(lin[1])
                print("The file has {} lines".format(lineCount) )
                print("Each Array is {} units long".format(arrSize) )
            # The first half of the arrays will be for training
            else:
                # We need to create a numpy array with the values in the line
                arr = []
                for ind in range(0,arrSize):
                    # make sure to convert each number to an integer
                    arr.append([ int( lin[ind] ) ])
                vect = np.array(arr)
                
                expectLet = lin[arrSize]
                expectVal = 0 if (expectLet=='I') else 1

                # create the object of type InputVect
                # The first half of these InputVects will be for training
                # The second half will be for testing
                input_vect = InputVect(vect, expectLet, expectVal)
                if(currentLine <= (lineCount/2) ):
                    train_ins.append(input_vect)
                else:
                    test_ins.append(input_vect)


            # Go to next Line
            currentLine += 1
    
    
    print("these are the training vectors and their expectations")
    for train_in in train_ins:
        print("{} .... {}".format(train_in.elements.transpose() , train_in.expectLet) )
    
    #------------------- training time ---------------------------------
    trainer = NeuroTraining(train_ins, nero)
    trainer.single_round()
    trainer.single_round()
    nero = trainer.get_trained_neuron()

    print("these are the testing vectors")
    for test_in in test_ins:
        print(test_in.elements.transpose() )

    #------------------ Testing Time ---------------------------------
    tester = NeuroTesting(test_ins, nero)
    tester.single_round()


#!!!!! comment out whichever one will not be used !!!!!

# for 3x3 inputs
training_func('threedimtest.csv', [[1],[0],[1],[0],[1],[0],[1],[0],[1]])

# for 5x5 inputs
#training_func('fivedimtest.csv',[[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1]] )
