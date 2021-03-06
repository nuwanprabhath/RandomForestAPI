import os
import sys

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.abspath('..'))
from RandomForestCore import forest  # Should be placed after path set.

app = Flask(__name__)


@app.route('/')
def index():
    return "Welcome to forest!"


@app.route('/predictProbAll', methods=['POST'])
def predict_prob_post_all():
    keys = request.json
    if bool(keys):
        print "keys in predictProbAll", keys
        response = forest.predict_number_prob_all(keys)
        print 'Sending response in predictProbAll: ' + str(response)
        return jsonify(response), 200
    else:
        return jsonify("Bad request. Json is empty"), 400


@app.route('/predictProb', methods=['POST'])
def predict_prob_post():
    keys = request.json
    if bool(keys):
        print "keys in predictProb", keys
        response = forest.predict_number_prob(keys)
        print 'Sending response in predictProb: ' + str(response)
        return jsonify(response), 200
    else:
        return jsonify("Bad request. Json is empty"), 400


@app.route('/predict', methods=['POST'])
def predict_post():
    keys = request.json
    if bool(keys):
        print "keys in predict", keys
        response = forest.predict_number(keys)
        print 'Sending response in predict: ' + str(response)
        return jsonify(response), 200
    else:
        return jsonify("Bad request. Json is empty"), 400


@app.route('/train', methods=['POST'])
def train():
    keys = request.json
    if bool(keys):
        print "keys in predict", keys
        response = forest.train(keys)
        print "Sending response in predict", response
        return jsonify(response), 200
    else:
        return jsonify("Bad request. Json is empty"), 400


@app.route('/resetCsv', methods=['GET'])
def reset_csv():
    print "keys in predict"
    response = forest.reset_csv()
    print "Sending response in resetCsv", response
    return jsonify(response), 200


if __name__ == '__main__':
    app.config.from_pyfile('../config.py')
    app.run('0.0.0.0', port=app.config['PORT'])
