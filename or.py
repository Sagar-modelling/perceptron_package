from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np

def main(data, eta, epochs, filename, plotfilename):
     
      df = pd.DataFrame(data)
      print(df)

      X,y = prepare_data(df)
      model_OR= Perceptron(eta=eta, epochs=epochs)
      model_OR.fit(X, y)

      _ = model_OR.total_loss()
      save_model(model_OR,filename=filename)
      save_plot(df,plotfilename, model_OR)

if __name__ == '__main__': #entry point

      OR = {"x1" : [0,0,1,1],
            "x2" : [0,1,0,1],
            "y"  : [0,1,1,1]
            }
      
      LR = 0.3
      EPOCHS = 4

      main(data=OR, eta=LR, epochs=EPOCHS, filename="or.model", plotfilename="or.png")