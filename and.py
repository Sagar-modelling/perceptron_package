"""
Author: Sagar
email: sagariit.kanpur1@gmail.com
"""

from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: - %(levelname)s: %(module)s: %(message)s]"
logging_dir = "logs"
os.makedirs(logging_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(logging_dir,"Running_logs.log"),level=logging.INFO, format=logging_str,
filemode='a')

def main(data, eta, epochs, filename, plotfilename):
     
      df = pd.DataFrame(data)
      logging.info(f"This is the actual dataframe{df}")
      X,y = prepare_data(df)
      model = Perceptron(eta=eta, epochs=epochs)
      model.fit(X, y)

      _ = model.total_loss()
      save_model(model,filename=filename)
      save_plot(df,plotfilename, model)

if __name__ == '__main__': #entry point

      AND = {
      "x1" : [0,0,1,1],
      "x2" : [0,1,0,1],
      "y" : [0,0,0,1],
      }
      
      LR = 0.3
      EPOCHS = 4  
      try:
            logging.info(">>>>>>>>>>>> starting training >>>>>>>>>>>>>")
            main(data=AND, eta=LR, epochs=EPOCHS, filename="and.model", plotfilename="and.png")
            logging.info(">>>>>>>>>>>> training done successfully >>>>>>>>>>>>>\n")
      except Exception as e:
            logging.exception(e)
            raise e

