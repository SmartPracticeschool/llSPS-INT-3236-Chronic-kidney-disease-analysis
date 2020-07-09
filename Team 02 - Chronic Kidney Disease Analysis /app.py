from flask import Flask, request, jsonify, render_template

from joblib import load

app = Flask(__name__)
model = load('Model Files/kidneysvm.save')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/y_predict', methods=['POST'])
def y_predict():
    x_test = [[float(x) for x in request.form.values()]]
    print(x_test)
    sc = load('Model Files/xtransform.save')
    prediction = model.predict(sc.transform(x_test))
    print(prediction)
    output = prediction[0]
    if output == 0:
        pred = "This person is likely to have CKD"
    else:
        pred = "This person is likely not to have CKD"

    return render_template('index.html', prediction_text=pred)


if __name__ == "__main__":
    app.run(debug=True)
