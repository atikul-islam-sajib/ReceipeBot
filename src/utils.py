import yaml
import joblib


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)

    else:
        raise Exception("value and filename must be provided".capitalize())


def load(filename=None):
    if filename is not None:
        return joblib.load(filename=filename)

    else:
        raise Exception("Filename should be passed".capitalize())


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)
