from flask import Flask
import flask
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import pickle
from sklearn import tree
from io import StringIO
from flask import request
import os
import json

app = Flask(__name__)

# Uncomment below code, If application running in Linux/MacOS
#base_path = "/root/"

# Uncomment below code, If application running in Windows
base_path = 'c:\\MY_WORK\\'
model_path = os.path.join(base_path, 'ml')
file_name = 'finalized_model.sav'


def model_trainer():
    print('Model training started, Please wait this will take a while...')
    X_train, X_test, Y_train, Y_test = split_data()

    # Fit the model on training set
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    score = accuracy_score(Y_test, prediction)
    print("Training SCORE ", score)
    # save the model to disk
    pickle.dump(clf, open(os.path.join(model_path, file_name), 'wb'))

    return 'Model is trained with {} Score'.format(score)


@app.route('/invoke', methods=['POST'])
def invoke_api():
    datas = None
    score = None
    prediction = None
    response = None
    dic = None
    try:

        if request.content_type == 'text/csv':
            data = request.data.decode('utf-8')
            s = StringIO(data)
            datas = pd.read_csv(s, header=None, error_bad_lines=False, skiprows=3, skipfooter=1)
        else:
            return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

        # load the model from disk
        loaded_model = pickle.load(open(os.path.join(model_path, file_name), 'rb'))
        try:
            prediction = loaded_model.predict(datas)
        except Exception as ee:
            print("EE", ee)
        dic = dict()
        if int(prediction[0]) == 1:
            dic['Message'] = "This person is likley to have type 2 diabetes"
        else:
            dic['Message'] = "Not a diabetes person"

        response = json.dumps(dic)
    except FileNotFoundError as e:
        print("ERROR ", e)
        return 'It seems model has not been trained, <br /><br /> <b>For training the model</b> <a ' \
               'href="http://localhost:5000/trainModel">Click Me</a> '
    except Exception as ae:
        print("Exception ", ae)
        return 'Please contact to admin for resolving current issues, Thanks'
    finally:
        dic = None

    return response


@app.route('/score', methods=['POST'])
def get_score():
    prediction = None
    data = request.get_json()
    dic = None

    df = pd.DataFrame([x.split(',') for x in data['diabetic_record'].split('\n')])
    loaded_model = pickle.load(open(os.path.join(model_path, file_name), 'rb'))
    try:
        prediction = loaded_model.predict(df)
    except Exception as ee:
        print("EE", ee)

    dic = dict()

    if int(prediction[0]) == 1:
        dic['Message'] = "This person is likley to have type 2 diabetes"
    else:
        dic['Message'] = "Not a diabetes person"

    return json.dumps(dic)


def split_data(test_size=0.33):
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(url, names=names)
    array = dataframe.values
    X = array[:, 0:8]
    Y = array[:, 8]
    seed = 7
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
    return X_train, X_test, Y_train, Y_test


@app.route("/")
def welcome_page():
    response = 'Hi, for getting score please apply "<b>/score</b>" post URL <br /><br />' \
               ' Apply JSON format as given below : <br /><br />' \
               ' {"diabetic_record":"2,197,70,45,543,30.5,0.158,53"}<br /><br />' \
               'All the input values are based on <br />' \
               ' <b color="green">["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age"] </b> and ' \
               '<b color="blue">Hash Diabetes</b> is response based on given input' \

    return response;


if __name__ == "__main__":
    output = model_trainer()
    print(output)
    app.run(host='0.0.0.0')
