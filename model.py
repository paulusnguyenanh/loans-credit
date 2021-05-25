
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle

data = pd.read_csv('./data/dataset.csv').values
N, d = data.shape
x = data[:, 0:d-1]
y = data[:, 2]

# su dung logistic regression tu thu vien sklearn

logreg = linear_model.LogisticRegression()
logreg.fit(x,y)
# luu cac bien vao mo hinh

pickle.dump(logreg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
a = [[5, 2]]
print(model.predict(a))
