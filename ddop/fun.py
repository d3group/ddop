from sklearn import datasets
import pandas as pd

def say_hello(name):
    return print("Hello " + name)


def load_iris_data():
    data = datasets.load_iris()
    return(data)


def load_my_data():
    df = pd.read_csv("../yaz_steak.csv")
    return(df)


