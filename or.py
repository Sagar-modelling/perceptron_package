from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np


OR = {"x1" : [0,0,1,1],
      "x2" : [0,1,0,1],
      "y"  : [0,1,1,1]
}
df = pd.DataFrame(OR)
print(df)

X,y = prepare_data(df)

LR = 0.3
EPOCHS = 5

model_OR = Perceptron(lr=LR, epochs=EPOCHS)
model_OR.fit(X, y)

_ = model_OR.total_loss()

save_model(model_OR,filename="or.model")
save_plot(df,"or.png",model_OR)