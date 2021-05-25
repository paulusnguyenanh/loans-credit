import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    output = int(output)
    if output == 1:
        daura = " Chúc mừng bạn có thẻ vay"
    else:
        daura = " Tiếc quá. Bạn không đủ điều kiện vay rồi"

    return render_template('index.html', daura=daura)


if __name__ == "__main__":
    app.run(debug=True)