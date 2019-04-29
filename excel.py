import xlrd

loc = ("C:\\Users\\Mahmoud\\Downloads\\4th Year 2nd Term\\Project\\Last updated.xlsx")

wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)


x0 =[]
x1 =[]
x2 =[]
x3 =[]
x4 =[]
x5 =[]
x6 =[]
x7 =[]
x8 =[]
x9 =[]
x10 =[]
x11 =[]


for i in range(sheet.nrows):
    if(i>1):
        x0.append(sheet.cell_value(i, 0))
        x1.append(sheet.cell_value(i, 1))
        x2.append(sheet.cell_value(i, 2))
        x3.append(sheet.cell_value(i, 3))
        x4.append(sheet.cell_value(i, 4))
        x5.append(sheet.cell_value(i, 5))
        x6.append(sheet.cell_value(i, 6))
        x7.append(sheet.cell_value(i, 7))
        x8.append(sheet.cell_value(i, 8))
        x9.append(sheet.cell_value(i, 9))
        x10.append(sheet.cell_value(i, 10))
        x11.append(sheet.cell_value(i, 11))


print("this is")
print(x0.__len__())
print(x1.__len__())
print(x2.__len__())
print(x3.__len__())
print(x4.__len__())
print(x5.__len__())
print(x6.__len__())
print(x7.__len__())
print(x8.__len__())
print(x9.__len__())
print(x10.__len__())
print(x11.__len__())





"""
training_inputs = np.array([[4, 64, 6.3,9],
                                [4,64,6.4,9],
                                [1, 0, 1,9],
                                [3,32,6.3,8.1]])

training_outputs = np.array([[.94, .92, .88, .87]]).T
"""

"""


    for i in range(sheet.nrows):
        if (i > 1):
            x[0].append(sheet.cell_value(i, 0))
            x[1].append(sheet.cell_value(i, 1))
            x[2].append(sheet.cell_value(i, 2))
            x[3].append(sheet.cell_value(i, 3))
            x[4].append(sheet.cell_value(i, 4))
            x[5].append(sheet.cell_value(i, 5))
            x[6].append(sheet.cell_value(i, 6))
            x[7].append(sheet.cell_value(i, 7))
            x[8].append(sheet.cell_value(i, 8))
            x[9].append(sheet.cell_value(i, 9))
            x[10].append(sheet.cell_value(i, 10))
            x[11].append(sheet.cell_value(i, 11))


    List1 =[]
    List2 = []
    List1 = [[x[0][0], x[1][0], x[2][0], x[3][0] ,x[4][0] ,x[5][0] ,x[6][0] ,x[7][0] ,x[8][0] ,x[9][0] ,x[10][0]]]
    for i in range(sheet.nrows-2):
        List1.append([x[0][i], x[1][i], x[2][i], x[3][i] ,x[4][i] ,x[5][i] ,x[6][i] ,x[7][i] ,x[8][i] ,x[9][i] ,x[10][i]])
        List2.append(x[11][i])

    training_inputs=np.array(List1)
    print("List1")
    print(List1)
    print(" inputs ")
    print(training_inputs)
    print("List2")
    print(List2)
    training_outputs = np.array(List2).T
    print(training_outputs)


 print("outSide")
    print(training_inputs.__len__())
    print(training_outputs.__len__())
    print(training_inputs)
    print(training_outputs)




"""


"""    l=[[1,2],[3,4]]
    print(l)
    l.append([5,6])
    print("cdcd")
    print(l)
"""






#
# import numpy as np
#
#
# class NeuralNetwork():
#
#     def __init__(self):
#         # seeding for random number generation
#         np.random.seed(1)
#
#         # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
#         self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
#
#     def sigmoid(self, x):
#         # applying the sigmoid function
#         return 1 / (1 + np.exp(-x))
#
#     def sigmoid_derivative(self, x):
#         # computing derivative to the Sigmoid function
#         return x * (1 - x)
#
#     def train(self, training_inputs, training_outputs, training_iterations):
#         # training the model to make accurate predictions while adjusting weights continually
#         for iteration in range(training_iterations):
#             # siphon the training data via  the neuron
#             output = self.think(training_inputs)
#
#             # computing error rate for back-propagation
#             error = training_outputs - output
#
#             # performing weight adjustments
#             adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
#
#             self.synaptic_weights += adjustments
#
#     def think(self, inputs):
#         # passing the inputs via the neuron to get output
#         # converting values to floats
#
#         inputs = inputs.astype(float)
#         output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
#         return output
#
#
# if __name__ == "__main__":
#     # initializing the neuron class
#     neural_network = NeuralNetwork()
#
#     print("Beginning Randomly Generated Weights: ")
#     print(neural_network.synaptic_weights)
#
#     # training data consisting of 4 examples--3 input values and 1 output
#     training_inputs = np.array([[0, 0, 1],
#                                 [1, 1, 1],
#                                 [1, 0, 1],
#                                 [0, 1, 1]])
#
#     training_outputs = np.array([[0, 1, 1, 0]]).T
#
#     # training taking place
#     neural_network.train(training_inputs, training_outputs, 15000)
#
#     print("Ending Weights After Training: ")
#     print(neural_network.synaptic_weights)
#
#     user_input_one = str(input("User Input One: "))
#     user_input_two = str(input("User Input Two: "))
#     user_input_three = str(input("User Input Three: "))
#
#     print("Considering New Situation: ", user_input_one, user_input_two, user_input_three)
#     print("New Output data: ")
#     print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
#     print("Wow, we did it!")
#













# import numpy as np
# #import tensorflow as tf
# import decimal
# from scipy.special import expit
# import xlrd
#
# class NeuralNetwork():
#
#     def __init__(self):
#         # seeding for random number generation
#         np.random.seed(1)
#
#         # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
#         self.synaptic_weights = 2 * np.random.random((11, 1)) - 1
#
#     def sigmoid(self, x):
#         # applying the sigmoid function
#         # x = expit(x)                          # expite : make the large number in the denominator equal to 0
#
#         #decimal.getcontext().prec = 100
#         #ccd = np.asarray([decimal.Decimal(el) for el in x], dtype=object)
#         #decimal_x = [[decimal.Decimal(z) for z in y] for y in x]
#         #ss = np.array(decimal_x)
#         return 1 / (1 + np.exp(-x))
#
#     def sigmoid_derivative(self, x):
#         # computing derivative to the Sigmoid function
#         return x * (1 - x)
#
#     def train(self, training_inputs, training_outputs, training_iterations):
#         # training the model to make accurate predictions while adjusting weights continually
#         for iteration in range(training_iterations):
#             # siphon the training data via  the neuron
#             output = self.think(training_inputs)           # isnot should define the index of the training_inputs that want to think of it beacuse now he think of all the training inputs how then ??
#
#             # computing error rate for back-propagation
#             error = training_outputs - output
#             # print("training_outputs  = " + str(training_outputs))
#             # print("output  = " + str(output))
#             # print("error  = " + str(error))
#             # performing weight adjustments
#             adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
#
#             self.synaptic_weights += adjustments
#         print("error  = "+str(error))
#
#     def think(self, inputs):
#         # passing the inputs via the neuron to get output
#         # converting values to floats
#
#         inputs = inputs.astype(float)
#         output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
#         return output
#
#
#
# def excelReading():
#
#     loc = ("C:\\Users\\Mahmoud\\Downloads\\4th Year 2nd Term\\Project\\Last updated.xlsx")
#     wb = xlrd.open_workbook(loc)
#     sheet = wb.sheet_by_index(0)
#
#     x = [[], [], [], [], [], [], [], [], [], [], [], []]
#
#     for i in range(sheet.nrows):
#         if (i > 1):
#             x[0].append(float(sheet.cell_value(i, 0)))
#             x[1].append(float(sheet.cell_value(i, 1)))
#             x[2].append(float(sheet.cell_value(i, 2)))
#             x[3].append(float(sheet.cell_value(i, 3)))
#             x[4].append(float(sheet.cell_value(i, 4)))
#             x[5].append(float(sheet.cell_value(i, 5)))
#             x[6].append(float(sheet.cell_value(i, 6)))
#             x[7].append(float(sheet.cell_value(i, 7)))
#             x[8].append(float(sheet.cell_value(i, 8)))
#             x[9].append(float(sheet.cell_value(i, 9)))
#             x[10].append(float(sheet.cell_value(i, 10)))
#             x[11].append(float(sheet.cell_value(i, 11)))
#
#     List1 = []
#     List2 = []
#     for i in range(sheet.nrows - 2):
#         List1.append(
#             [x[0][i], x[1][i], x[2][i], x[3][i], x[4][i], x[5][i], x[6][i], x[7][i], x[8][i], x[9][i], x[10][i]])
#         List2.append(x[11][i])
#
#     training_inputs = np.array(List1)
#     training_outputs = np.array([List2]).T
#     print("inside")
#     print(training_inputs )
#     print(training_outputs)
#     print(training_inputs.shape)
#     print(training_outputs.shape)
#
#     return (training_inputs , training_outputs)
#
#
#
# if __name__ == "__main__":
#     # initializing the neuron class
#     neural_network = NeuralNetwork()
#
#     print("Beginning Randomly Generated Weights: ")
#     print(neural_network.synaptic_weights)
#     print("shape")
#     print(neural_network.synaptic_weights.shape)
#     # training data consisting of 4 examples--3 input values and 1 output
#
#
#     training_inputs , training_outputs = excelReading()
#     print("::::::::::::")
#     print(type(training_inputs[0][0]))
#
#     # training taking place
#     neural_network.train(training_inputs, training_outputs, 15000)
#
#     print("Ending Weights After Training: ")
#     print(neural_network.synaptic_weights)
#
#     print("Enter the attributes ")
#     uInputs = []
#
#     for i in range(11):
#         uInputs.append(input())
#
#     # user_input_one = str(input("User Input One: "))
#     # user_input_two = str(input("User Input Two: "))
#     # user_input_three = str(input("User Input Three: "))
#     # user_input_four = str(input("User Input Four: "))
#     #print("Considering New Situation: ", user_input_one, user_input_two, user_input_three ,user_input_four)
#     print("Considering New Situation: ", uInputs)
#     print("New Output data: ")
#     #print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three , user_input_four])))
#     print(neural_network.think(np.array([uInputs])))
#     print("Wow, we did it!")
#








