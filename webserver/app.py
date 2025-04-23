from flask import Flask
from flask import request
from flask import render_template

import os
import json


class Model:

    def __init__(self):
        with open('assets/phrase.txt', 'r') as file:
            self.phrase = file.read()

    def predict(self, input):
        return self.phrase


model = Model()

app = Flask(__name__)
app.model = model


@app.route('/', methods=['GET'])
def index():
    user_input = request.args.get(
        'input',
        default='',
        type=str,
    )
    prediction = app.model.predict(user_input)
    output = {
        'input': user_input,
        'prediction': prediction,
    }
    return app.response_class(
        response=json.dumps(output),
        status=200,
        mimetype='application/json',
    )


def run():
    if not os.path.exists(app.static_folder):
        print(f"WARNING: static folder {app.static_folder} not found")
        print("Current working directory:", os.getcwd())
        print("Directory contents:", os.listdir(os.getcwd()))

    app.run(debug=True)


if __name__ == "__main__":
    run()
