import numpy as np 

class Perceptron:
  def __init__(self,eta,epochs):
    np.random.seed(42)
    self.weights = np.random.randn(3) * 10**-4
    print(f"intial weights brfore training: {self.weights}")
    self.eta = eta
    self.epochs = epochs    #self declares variable as global for all the methods

  def activationFunction(self,inputs,weights): 
    z = np.dot(inputs,weights)
    return np.where(z>0, 1,0)
  
  def fit(self, X, y):
    self.X = X 
    self.y = y

    X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))] #concatenation of X and bias 
    print(f"X with bias: {X_with_bias}")

    for epoch in range(self.epochs):
      print("--"*10)
      print(f"for epoch: {epoch}")
      print("--"*10)

      y_hat = self.activationFunction(X_with_bias, self.weights) #forward propogation
      print(f"prdicted value after forward pass: \n{y_hat}")
      self.error = self.y - y_hat
      print(f"error : \n{self.error}")
      self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error) #backward propogation
      print(f"updated weights after epoch : \n{epoch}/{self.epochs} : \n{self.weights}")
      print("######"*10)

  def predict(self, X):
    X_with_bias = np.c_[X, -np.ones((len(X), 1))]
    return self.activationFunction(X_with_bias, self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"total loss : {total_loss}")
    return total_loss
