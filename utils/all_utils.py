import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib #for saving model as a binary file
from matplotlib.colors import ListedColormap
plt.style.use("fivethirtyeight") #style of graphs
import os

def prepare_data(df):
  """It is used to seperate the dependent and independent features

  Args:
      df (pd:DataFrame): Its is pandas dataframe

  Returns:
      tuple: It returns tuples of dependent and Independent Variables
  """

  X = df.drop("y", axis=1)

  y= df["y"]
  return X,y

def save_model(model,filename):
  """This saves the trained model 

  Args:
      model (python object): trained model
      filename (str): path to save the trained model
  """

  model_dir  = "models"
  os.makedirs(model_dir, exist_ok=True)
  filepath = os.path.join(model_dir, filename)#models/filename
  joblib.dump(model, filepath)

def save_plot(df, file_name, model):
  """
  :param df: its a dataframe object
  :param file_name: its the path to save the plot
  :param model: passing the model

  """

  def _create_base_plot(df):  #internal function
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color = "black", linestyle="--",linewidth = 1)
    plt.axvline(x=0, color = "black", linestyle="--",linewidth = 1)
    figure = plt.gcf() #get current figure
    figure.set_size_inches(10,8)

  def _plot_decision_region(X,y,classifier,resolution=0.02):  #internal function
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    X = X.values #X as an array
    x1 = X[:,0]
    x2 = X[:,1]
    x1_min, x1_max = x1.min() - 1 , x1.max() + 1 #finding minimum and ,maximum value in the x1[0] and x2[1] column
    x2_min, x2_max = x2.min() - 1 , x2.max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    print(xx1)
    print(xx1.ravel())
    
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2, Z, alpha =0.2, cmap=cmap) #ALPHA DENOTES TRANSPARENCY
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot()

  X,y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_region(X,y,model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True)
  plotpath = os.path.join(plot_dir, file_name)   #plots/filename
  plt.savefig(plotpath)