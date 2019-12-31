from flask import Flask
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)


@app.route("/trainModel")
def load_pickel_file():
    X_train, X_test, Y_train, Y_test = train_model()
    # Fit the model on training set
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    return 'Model is trained now.. Please follow below link to get result <br /> <br /> <a ' \
           'href="http://localhost:5000/getResult">Get Result</a> '


@app.route("/getResult")
def get_result():
    try:
        X_train, X_test, Y_train, Y_test = train_model()
        filename = 'finalized_model.sav'
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.score(X_test, Y_test)
        print(result)
    except FileNotFoundError:
        return 'It seems model has not been trained, <br /><br /> <b>For training the model</b> <a ' \
               'href="http://localhost:5000/trainModel">Click Me</a> '

    return str(result)


def train_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pandas.read_csv(url, names=names)
    array = dataframe.values
    X = array[:, 0:8]
    Y = array[:, 8]
    test_size = 0.33
    seed = 7
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

    return X_train, X_test, Y_train, Y_test


@app.route("/")
def welcome_page():
    response = 'Follow below instruction to train the model and getting the result<br /><br /> 1). For training the ' \
               'model click here <a href="http://localhost:5000/trainModel">Train Model</a> <br /><br /> 2). For ' \
               'getting result click here <a href="http://localhost:5000/getResult">Get Result</a> '
    return response;


if __name__ == "__main__":
    app.run(host='0.0.0.0')
