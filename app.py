from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# model = pickle.load(open('final_model.pkl', 'rb'))
with open('final_model.pkl', 'rb') as model_file:
  loaded_objects = pickle.load(model_file)

model = loaded_objects["model"]
stand_scaler = loaded_objects["scaler"]


@app.route("/")
def hello():
  return render_template('home.html')


@app.route("/predict", methods=['POST'])
def predict():
  no_of_dependents = float(request.form.get('no_of_dependents'))
  education = float(request.form.get('education'))
  self_employed = float(request.form.get('self_employed'))
  income_annum = float(request.form.get('income_annum'))
  loan_amount = float(request.form.get('loan_amount'))
  loan_term = float(request.form.get('loan_term'))
  cibil_score = float(request.form.get('cibil_score'))
  residential_assets_value = float(
    request.form.get('residential_assets_value'))
  commercial_assets_value = float(request.form.get('commercial_assets_value'))
  luxury_assets_value = float(request.form.get('luxury_assets_value'))
  bank_asset_value = float(request.form.get('bank_asset_value'))

  input_data = [
    no_of_dependents, income_annum, loan_amount, loan_term, cibil_score,
    residential_assets_value, commercial_assets_value, luxury_assets_value,
    bank_asset_value, education, self_employed
  ]

  input_data_as_numpy_array = np.array(input_data).reshape(1, -1)

  # Transform the input data using the fitted scaler
  final = stand_scaler.transform(input_data_as_numpy_array)

  prediction = model.predict(final)

  if (prediction[0] == 1):
    return render_template('prediction_yes.html')
  else:
    return render_template('prediction_no.html')


@app.route("/form_info", methods=['GET', 'POST'])
def form_info():
  answers = dict(request.form)
  return jsonify(answers)


if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True)
