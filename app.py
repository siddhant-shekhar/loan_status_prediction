from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import sklearn

app = Flask(__name__)

model = pickle.load(open('final_model.pkl', 'rb'))


@app.route("/")
def hello():
  return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
  no_of_dependents = (request.form.get('no_of_dependents'))
  education = (request.form.get('education'))
  self_employed = (request.form.get('self_employed'))
  income_annum = (request.form.get('income_annum'))
  loan_amount = (request.form.get('loan_amount'))
  loan_term = (request.form.get('loan_term'))
  cibil_score = (request.form.get('cibil_score'))
  residential_assets_value = (request.form.get('residential_assets_value'))
  commercial_assets_value = (request.form.get('commercial_assets_value'))
  luxury_assets_value = (request.form.get('luxury_assets_value'))
  bank_asset_value = (request.form.get('bank_asset_value'))

  prediction = model.predict(
    np.asarray([
      no_of_dependents, education, self_employed, income_annum, loan_amount,
      loan_term, cibil_score, residential_assets_value,
      commercial_assets_value, luxury_assets_value, bank_asset_value
    ]).reshape(1, -1))

  if (prediction[0] == 1):
    return render_template('prediction_no.html')
  else:
    return render_template('prediction_yes.html')


@app.route("/form_info", methods=['GET', 'POST'])
def form_info():
  answers = dict(request.form)
  return jsonify(answers)


if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True)
