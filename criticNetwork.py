from parameters import Parameters as p, DiamondBoard as db, TriangleBoard as tb
import numpy as np
import torch


def stringToList(state):  # Turns string state into input layer
    out = []
    for s in state:
        out.append(int(s))
    return out


class NetworkCritic(torch.nn.Module):

    def __init__(self, learningRate, discFactor, traceDecay, architecture):
        super(NetworkCritic, self).__init__()
        self.weights = torch.nn.ModuleList()
        for num, val in enumerate(architecture):
            if num == 0:
                continue
            self.weights.append(torch.nn.Linear(architecture[num - 1], val))
        self.learningRate = learningRate
        self.discFactor = discFactor
        self.traceDecay = traceDecay
        self.eTrace = []
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learningRate, momentum=0.0)

    def resetETrace(self):
        self.eTrace = []
        for layer in self.parameters():
            self.eTrace.append(torch.Tensor(np.zeros(tuple(layer.shape))))

    def setETrace(self, state):
        pass

    def updateETrace(self, state):
        pass

    def setExpValues(self, state):
        pass

    def getExpValues(self, state):
        return self(torch.Tensor(stringToList(state)))

    def updateExpValues(self, state, TDError, isCurrentState=False):
        if isCurrentState:
            self.optimizer.zero_grad()
            outputs = self(torch.Tensor(stringToList(state)))
            MSELoss = self.criterion(outputs + TDError, outputs)
            MSELoss.backward(retain_graph=True)
            paramNumber = 0
            for param in self.parameters():
                self.eTrace[paramNumber] = self.eTrace[paramNumber] + param.grad * ((2 * float(TDError)) ** (-1))
                param.grad = float(TDError) * self.eTrace[paramNumber]
                self.eTrace[paramNumber] = self.discFactor * self.traceDecay * self.eTrace[paramNumber]
                paramNumber += 1
            self.optimizer.step()

    def forward(self, inputLayer):
        layerNumber = 0
        for layer in self.weights:
            layerNumber += 1
            if layerNumber == len(self.weights):  # If last layer (output layer)
                inputLayer = torch.tanh(layer(inputLayer))
            else:
                inputLayer = torch.nn.functional.relu(layer(inputLayer))
        return inputLayer
