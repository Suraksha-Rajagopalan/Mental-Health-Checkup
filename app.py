import pandas as pd
import joblib
from flask import Flask, request, jsonify,  render_template, redirect, url_for, session, flash
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired, Email, ValidationError
import bcrypt
from flask_mysqldb import MySQL
from wtforms import StringField,PasswordField,SubmitField

app = Flask(__name__, template_folder='templates')

# Load the pre-trained model
model_path = 'D:/MINU/Projects/mental_health/best_random_forest_model.pkl'  
model = joblib.load(model_path)

# Load the CSV file with medians/modes based on age groups
age_group_file = 'D:/MINU/Projects/mental_health/age_group_medians_modes.csv'
age_group_df = pd.read_csv(age_group_file)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'mydatabase'
app.secret_key = 'your_secret_key_here'

mysql = MySQL(app)

class RegisterForm(FlaskForm):
    name = StringField("Name",validators=[DataRequired()])
    email = StringField("Email",validators=[DataRequired(), Email()])
    password = PasswordField("Password",validators=[DataRequired()])
    submit = SubmitField("Register")

    def validate_email(self,field):
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users where email=%s",(field.data,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            raise ValidationError('Email Already Taken')

class LoginForm(FlaskForm):
    email = StringField("Email",validators=[DataRequired(), Email()])
    password = PasswordField("Password",validators=[DataRequired()])
    submit = SubmitField("Login")


#Hopme Page
@app.route('/')
def index():
    return render_template('index.html')

#Register
@app.route('/register',methods=['GET','POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        name = form.name.data
        email = form.email.data
        password = form.password.data

        hashed_password = bcrypt.hashpw(password.encode('utf-8'),bcrypt.gensalt())

        # store data into database 
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users (name,email,password) VALUES (%s,%s,%s)",(name,email,hashed_password))
        mysql.connection.commit()
        cursor.close()

        return redirect(url_for('login'))

    return render_template('register.html',form=form)

#Login
@app.route('/login',methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email=%s",(email,))
        user = cursor.fetchone()
        cursor.close()
        if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
            session['user_id'] = user[0]
            return redirect(url_for('dashboard'))
        else:
            flash("Login failed. Please check your email and password")
            return redirect(url_for('login'))

    return render_template('login.html',form=form)

# Dashboard
@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        user_id = session['user_id']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users where id=%s",(user_id,))
        user = cursor.fetchone()
        cursor.close()

        if user:
            return render_template('dashboard.html',user=user)
            
    return redirect(url_for('login'))

# Logout
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out successfully.")
    return redirect(url_for('login'))


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Map form data to a DataFrame
        data = {
            'Age': request.form['Age'],
            'City': request.form['City'],
            'Working Professional or Student': request.form['Work/Student'],
            'Profession': request.form['Profession'],
            'Academic Pressure': request.form['Academic Pressure'],
            'Work Pressure': request.form['Work Pressure'],
            'CGPA' : request.form['CGPA'],
            'Study Satisfaction': request.form['Study Satisfaction'],
            'Job Satisfaction': request.form['Job Satisfaction'],
            'Sleep Duration': request.form['Sleep Duration'],
            'Dietary Habits': request.form['Dietary Habits'],
            'Degree': request.form['Degree'],
            'Have you ever had suicidal thoughts ?': request.form['Suicidal Thoughts'],
            'Work/Study Hours': request.form['Work/Study Hours'],
            'Financial Stress': request.form['Financial Stress'],
            'Family History of Mental Illness': request.form['Family']
        }
        
        # Convert the incoming data into a DataFrame
        df = pd.DataFrame([data])

        # Ensure numeric columns are handled correctly
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['CGPA'] = pd.to_numeric(df['CGPA'], errors='coerce')
        df['Academic Pressure'] = pd.to_numeric(df['Academic Pressure'], errors='coerce')
        df['Work Pressure'] = pd.to_numeric(df['Work Pressure'], errors='coerce')
        df['Study Satisfaction'] = pd.to_numeric(df['Study Satisfaction'], errors='coerce')
        df['Job Satisfaction'] = pd.to_numeric(df['Job Satisfaction'], errors='coerce')
        df['Sleep Duration'] = pd.to_numeric(df['Sleep Duration'], errors='coerce')
        df['Work/Study Hours'] = pd.to_numeric(df['Work/Study Hours'], errors='coerce')
        df['Financial Stress'] = pd.to_numeric(df['Financial Stress'], errors='coerce')
        
        
        # Ensure numeric columns are handled correctly as integers
        numeric_cols = ['Age', 'CGPA', 'Academic Pressure', 'Work Pressure', 
                'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration', 
                'Work/Study Hours', 'Financial Stress']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1).astype('Int64')

        # Encode categorical features using LabelEncoder
        non_numeric_cols = df.select_dtypes(include=['object']).columns
        for col in non_numeric_cols:
            df[col] = label_encoder.fit_transform(df[col].astype(str))

        # Ensure all the required columns are present
        required_columns = ['Age', 'City', 'Working Professional or Student', 'Profession', 
                            'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 
                            'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 
                            'Degree', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 
                            'Financial Stress', 'Family History of Mental Illness']
        
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': 'Missing required columns in input data'}), 400

        # Make prediction
        df = df[required_columns]
        print(df.head())
        print(df.dtypes)

        prediction = model.predict(df.to_numpy())

        # Return prediction
        if prediction==0:
            return render_template("index.html", prediction_text = "The Predicted Result: Not Depressed.")
        else:
            return render_template("index.html", prediction_text = "The Predicted Result: Depressed.")

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
