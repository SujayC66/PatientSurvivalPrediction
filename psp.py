from flask import Flask,render_template,request,jsonify

import pickle
import numpy as np

with open ("ab_reg_hyp1.pkl","rb") as f:
    model=pickle.load(f)



app=Flask(__name__)
@app.route("/")
def index():
    return render_template("psp_data.html")

@app.route("/psp",methods=['POST'])
def predict():
    apache_4a_icu_death_prob = float(request.form['a'])
    apache_4a_hospital_death_prob  = float(request.form['b'])
    apache_3j_diagnosis  = float(request.form['c'])
    age  = float(request.form['d'])
    bmi  =float(request.form['e'])
    ventilated_apache  = float(request.form['f'])
    gender  = float(request.form['g'])
    apache_2_diagnosis  = float(request.form['h'])
    heart_rate_apache  = float(request.form['i'])
    d1_spo2_min  = float(request.form['j'])
    d1_heartrate_min  = float(request.form['k'])

    data = np.array([apache_4a_icu_death_prob,apache_4a_hospital_death_prob, apache_3j_diagnosis, age, bmi,ventilated_apache, gender,apache_2_diagnosis,
                  heart_rate_apache,d1_spo2_min,d1_heartrate_min],ndmin=2)
    result = model.predict(data)

    print(result)
    if result[0]==0:
        pred='dead'
    if result[0]==1:
        pred='survived'  

    return jsonify(pred) 

if __name__=="__main__":
    app.run(debug=True)