#import matplotlib.pyplot as plotter
import numpy as np

class InputVect(object):
    elements = []
    expect = None
    
    def __init__(self, els, exp):
        self.elements = els
        self.expect = exp

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
            print("\t\tI'll spit out a -1 ...")
            print("--------------------------------")
            return -1

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
    
    #inputs = []
    #nero = None
    total = 0
    tot_correct = 0
    
    def __init__(self, ins, n):
        #self.inputs = inps
        #self.nero = Neuron(init_weight, 0, 0.3)
        self.total = len(ins)
        self.tot_correct = 0
        NeuroActivity.__init__(self, ins, n)

    def single_round(self):
        for single_input in self.inputs:
            print("\n=================================================")
            print("Putting {} into nero... expecting {}".format(single_input.elements.transpose(), single_input.expect))
            
            output = self.nero.analyze_input(single_input.elements)
            difference = single_input.expect - output
        
            print("output is ... {}".format(output))
            if difference != 0:
                print("modifying nero's weights ... ...")
                self.nero.modify(single_input.elements, difference)
            else:
                self.tot_correct += 1
            
            print("=================================================\n")
    
    def train(self):
        #only stop training when all inputs get the correct result
        while self.tot_correct != self.total:
            self.tot_correct = 0
            print("NEW ROUND !!!!!!!!")
            self.single_round()

class NeuroTesting(NeuroActivity):

    def __init__(self, ins, n):
        NeuroActivity.__init__(self, ins, n)

    def single_round(self):
        for single_input in self.inputs:
            print("\n=================================================")
            print("Putting {} into nero... ".format(single_input.transpose() ) )

            output = self.nero.analyze_input(single_input)

            print("output is ... {}".format(output))
            print("=================================================\n")

#-------------------------------------------------------------------------------------------------------------------

def training_func():

    # This is Nero, say hi nero
    nero = Neuron( np.array([[0],[1],[0],[1],[0],[1]]) , 0, 0.3)

    #------------------ Training Setup -------------------------------
    
    # the array of the training inputs
    train_ins = []
    
    #here are the training inputs !!These are objects of type InputVector!!
    vect1 = np.array([[2],[4],[3],[-2],[-1],[0]])
    expect1 = -1
    input_vect1 = InputVect(vect1, expect1)
    train_ins.append(input_vect1)
    
    vect2 = np.array([[1],[0],[0],[4],[-1],[-2]])
    expect2 = 1
    input_vect2 = InputVect(vect2, expect2)
    train_ins.append(input_vect2)
    
    vect3 = np.array([[0.4],[0.6],[-0.7],[-0.2],[5],[-3]])
    expect3 = -1
    input_vect3 = InputVect(vect3, expect3)
    train_ins.append(input_vect3)

    vect4 = np.array([[3.14],[0.22],[0.02],[-0.02],[-0.22],[-3.14]])
    expect4 = 1
    input_vect4 = InputVect(vect4, expect4)
    train_ins.append(input_vect4) 
    
    print("these are the training vectors and their expectations")
    for every in train_ins:
        print(every.elements)
        print("-------------------------------------")
        print(every.expect)
        print(":0      :0      :0    \n\n")
        
    
    #------------------- training time ---------------------------------
    trainer = NeuroTraining(train_ins, nero)
    trainer.train()



    #----------------- Testing Setup ------------------------------------

    # the array of the test inputs
    test_ins = []

    # here are the testing inputs !!These are objects of type Numpy Array!!
    vectA = np.array([[1],[2],[3],[4],[5],[6]])
    test_ins.append(vectA)

    vectB = np.array([[-0.43],[-0.65],[-0.95],[0.25],[0.75],[0.24]])
    test_ins.append(vectB)

    vectC = np.array([[-6],[-5],[-4],[-3],[-2],[-1]])
    test_ins.append(vectC)

    vectD = np.array([[-1.55],[-2.33],[-0.01],[0.99],[0.98],[1.21]])
    test_ins.append(vectD)

    print("these are the testing vectors")
    for each in test_ins:
        print(each)
        print("-------------------------------------")

    #------------------ Testing Time ---------------------------------
    tester = NeuroTesting(test_ins, nero)
    tester.single_round()
            


training_func()


