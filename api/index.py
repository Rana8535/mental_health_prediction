import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
import os
from sqlalchemy.sql import func

basedir = os.path.abspath(os.path.dirname(__file__) + '/..')
app = Flask(__name__, template_folder='../templates', static_folder='../static')

app.config['SQLALCHEMY_DATABASE_URI'] =\
        'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class student(db.Model):
    id = db.Column(db.Integer, db.Sequence('seq_reg_id', start=1, increment=1),
               primary_key=True)
    q1=db.Column(db.Integer)
    q2=db.Column(db.Integer)
    q3=db.Column(db.Integer)
    q4=db.Column(db.Integer)
    q5=db.Column(db.Integer)
    q6=db.Column(db.Integer)
    q7=db.Column(db.Integer)
    q8=db.Column(db.Integer)
    q9=db.Column(db.Integer)
    q10=db.Column(db.Integer)
    q11=db.Column(db.Integer)
    q12=db.Column(db.Integer)
    q13=db.Column(db.Integer)
    q14=db.Column(db.Integer)
    q15=db.Column(db.Integer)
    q16=db.Column(db.Integer)
    q17=db.Column(db.Integer)
    q18=db.Column(db.Integer)
    q19=db.Column(db.Integer)
    q20=db.Column(db.Integer)
    q21=db.Column(db.Integer)
    q22=db.Column(db.Integer)
    q23=db.Column(db.Integer)
    q24=db.Column(db.Integer)
    q25=db.Column(db.Integer)
    q26=db.Column(db.Integer)
    q27=db.Column(db.Integer)
    prediction=db.Column(db.Integer)

#col_list=student.__table__.columns.keys()


@app.route('/')

def home():
    return render_template('index.html')

@app.route('/base')
def base():
    return render_template('base.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/service')
def service():
    context=['I am currently employed at least part-time', 'Education',
       'I have my own computer separate from a smart phone',
       'I have been hospitalized before for my mental illness',
       'How many days were you hospitalized for your mental illness',
       'I am legally disabled', 'I have my regular access to the internet',
       'I live with my parents', 'I have a gap in my resume',
       'Total length of any gaps in my resume in months.', 'Income',
       'Unemployed', 'I read outside of work and school',
       'Annual income from social welfare programs', 'I receive food stamps',
       'I am on section 8 housing',
       'How many times were you hospitalized for your mental illness',
       'Lack of concentration', 'Anxiety', 'Depression', 'Obsessive thinking',
       'Mood swings', 'Panic attacks', 'Compulsive behavior', 'Tiredness',
       'Age', 'Gender']
    return render_template('home.html',context=context)

@app.route('/team')
def team():
    return render_template('team.html')

def ValuePredictor(to_predict_list):
    to_predict = (to_predict_list)
    loaded_model = pickle.load(open("../model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/result1', methods = ['POST'])

def result():
    if request.method =='POST':

        to_predict_list = request.form.to_dict()

        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        to_predict_list = list(map(int, to_predict_list))
        #print(to_predict_list)
        r=to_predict_list
        for i in range(len(r)):
            r[i]=int(r[i])



        result = ValuePredictor([to_predict_list])
        data_lst=student(q1=r[0],q2=r[1],q3=r[2],q4=r[3],q5=r[4],q6=r[5],q7=r[6],q8=r[7],q9=r[8],q10=r[9],q11=r[10],q12=r[11],q13=r[12],q14=r[13],q15=r[14],q16=r[15],q17=r[16],q18=r[17],q19=r[18],q20=r[19],q21=r[20],q22=r[21],q23=r[22],q24=r[23],q25=r[24],q26=r[25],q27=r[26],prediction=result)
        print(data_lst)
        db.session.add(data_lst)
        db.session.commit()
        if int(result)== 1:
            prediction ='Prediction shows that you are  mentaly Ill'
        else:
            prediction ='Prediction shows that your are fine'
        return render_template("result1.html", prediction = prediction)
    else:
        return render_template("home.html")
@app.route('/Contact')
def Contact():
    return render_template('Contact.html')

@app.route('/Read_more')
def Read_more():
    return render_template('Read_more.html')


if __name__ == '__main__':
    app.run(debug=True)
