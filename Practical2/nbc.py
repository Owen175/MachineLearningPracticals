import numpy as np

class NBC:
    def __init__(self, feature_types, num_classes):
        self.featureTypes = feature_types
        self.numClasses = num_classes  # NOTE: May be higher than the number of classes in the training data.
        # Handled later. 
        self.pi = None  # The Nc/N
        self.Nc = None  # Number in each class - padded to handle above issue
        self.classes = None  # The classes by the name they are input in y_train/test
        self.classMeans = None  # Array of mean of each datapoint
        self.classVars = None  # Array of variance of each datapoint

    # Trains
    def fit(self, Xtrain, ytrain):
        inv = self.getPi(ytrain)
        self.getTheta(Xtrain, inv)


    # Finds the individual classes and the number in each.  
    def getPi(self, ytrain):
        self.classDict = {}
        self.classes, inverse, self.Nc = np.unique(ytrain, return_counts=True, return_inverse=True)
        for _ in range(self.numClasses-len(self.Nc)):
            self.Nc = np.append(self.Nc, 0) # Pads Nc and classes to deal with 0 of a class
            self.classes = np.append(self.classes, self.classes[0])
        self.pi = self.Nc / ytrain.size
        return inverse  # Inverse is the list of indices of classes to rebuild ytrain. 

    def getTheta(self, Xtrain, inv): 
        self.classMeans = np.empty((self.numClasses, len(self.featureTypes)))
        self.classVars = np.empty((self.numClasses, len(self.featureTypes)))
        for i in range(self.numClasses):
            classIdxs = inv == i  # Indexes at which class i is in the list. 
            if not np.any(classIdxs):  # Again to handle the case of an empty class. 
                self.classMeans[i] = 0
                self.classVars[i] = 0
                continue
            classIdxs = np.tile(classIdxs[:, np.newaxis], (1, len(self.featureTypes))) # Changes dimension to suit mean/var
            self.classMeans[i] = np.mean(Xtrain, axis=0, where=classIdxs)
            self.classVars[i] = np.var(Xtrain, axis=0, where=classIdxs)
        self.classVars[self.classVars < 1e-6] = 1e-6  # Ensures no error due to a too-small variance

    def predict(self, Xtest):
        # p(y|π)*p(x|y,θ)
        # Log space, so log (p(y|π)) + log (p(x|y,θ))
        # log p(x|y,θ) = sum of j=1 to D of p(xj|y=c,θjc)
        # Log of the gaussian pdf:
        # -0.5 * log(2pi) - 0.5 * log (var jc) - (xj-meanjc)squared/2varjc

        logProbxgiveny = np.array([-0.5 * np.sum(  # Corrects for later lack of negation and division by 2
                np.square(Xtest-self.classMeans[i])/self.classVars[i] +  # (xj-meanjc)^2/varjc
                    np.log(self.classVars[i]) + np.log(2 * np.pi),       # log var jc + log 2pi
                axis=1
            ) for i in range(self.numClasses)])
        # This is an array s.t. the rows are for each c value and the cols for each test value
        with np.errstate(divide='ignore'):  # If there is a 0 probability of one, it should be ignored. Makes no difference to end result
            resultProbs = logProbxgiveny + np.log(self.pi)[:, np.newaxis] # Second part of the formula - logs so adding. 
        return self.classes[np.argmax(resultProbs, axis=0)]  # Finds the class it is most likely to be
    
