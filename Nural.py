import numpy as np
#import tensorflow as tf
import decimal
from scipy.special import expit
import random
import xlrd

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        #self.synaptic_weights = 2 * np.random.random((11, 1)) - 1
        self.synaptic_weights = []
        for i in range(11) :
            sam = decimal.Decimal(random.randrange(-1,1)) / 100000
            self.synaptic_weights.append(sam)

        self.synaptic_weights = np.mat(self.synaptic_weights)
        self.synaptic_weights = self.synaptic_weights.reshape(11,1)

    def sigmoid(self, x):

        # x = expit(x)                          # expite : make the large number in the denominator equal to 0

        # decimal.getcontext().prec = 100
        # decimal_x = [decimal.Decimal(z) for z in x]
        # ss = np.array(decimal_x)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)           # isnot should define the index of the training_inputs that want to think of it beacuse now he think of all the training inputs how then ??

            # computing error rate for back-propagation
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments
        print("error  = "+str(error))

    def think(self, inputs):

        #inputs = inputs.astype(float)
        inputs = inputs.astype(decimal.Decimal)

        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output



def excelReading():

    loc = ("C:\\Users\\Mahmoud\\Downloads\\4th Year 2nd Term\\Project\\Last updated.xlsx")
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)

    x = [[], [], [], [], [], [], [], [], [], [], [], []]

    for i in range(sheet.nrows):
        if (i > 1):
            x[0].append(decimal.Decimal(sheet.cell_value(i, 0)))
            x[1].append(decimal.Decimal(sheet.cell_value(i, 1)))
            x[2].append(decimal.Decimal(sheet.cell_value(i, 2)))
            x[3].append(decimal.Decimal(sheet.cell_value(i, 3)))
            x[4].append(decimal.Decimal(sheet.cell_value(i, 4)))
            x[5].append(decimal.Decimal(sheet.cell_value(i, 5)))
            x[6].append(decimal.Decimal(sheet.cell_value(i, 6)))
            x[7].append(decimal.Decimal(sheet.cell_value(i, 7)))
            x[8].append(decimal.Decimal(sheet.cell_value(i, 8)))
            x[9].append(decimal.Decimal(sheet.cell_value(i, 9)))
            x[10].append(decimal.Decimal(sheet.cell_value(i, 10)))
            x[11].append(decimal.Decimal(sheet.cell_value(i, 11)))

    List1 = []
    List2 = []
    for i in range(sheet.nrows - 2):
        List1.append(
            [x[0][i], x[1][i], x[2][i], x[3][i], x[4][i], x[5][i], x[6][i], x[7][i], x[8][i], x[9][i], x[10][i]])
        List2.append(x[11][i])

    training_inputs = np.array(List1)
    training_outputs = np.array([List2]).T
    print("inside")
    print(training_inputs )
    print(training_outputs)
    print(training_inputs.shape)
    print(training_outputs.shape)

    return (training_inputs , training_outputs)



if __name__ == "__main__":
    # initializing the neuron class
    neural_network = NeuralNetwork()


    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)
    print("shape")
    print(neural_network.synaptic_weights.shape)
    print(neural_network.synaptic_weights)



    # training data consisting of 4 examples--3 input values and 1 output


    training_inputs , training_outputs = excelReading()
    print("::::::::::::")
    print(type(training_inputs[0][0]))

    # training taking place
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    print("Enter the attributes ")
    uInputs = []

    for i in range(11):
        uInputs.append(input())

    # user_input_one = str(input("User Input One: "))
    # user_input_two = str(input("User Input Two: "))
    # user_input_three = str(input("User Input Three: "))
    # user_input_four = str(input("User Input Four: "))
    #print("Considering New Situation: ", user_input_one, user_input_two, user_input_three ,user_input_four)
    print("Considering New Situation: ", uInputs)
    print("New Output data: ")
    #print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three , user_input_four])))
    print(neural_network.think(np.array([uInputs])))
    print("Wow, we did it!")

















# print("decimal")
    # print(float(14544.1515165))
    # print(decimal.Decimal(14544.1515165))



