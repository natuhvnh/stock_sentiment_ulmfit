import numpy as np
from flask import Flask, request, jsonify
from flask_restplus import Api, Resource, fields
from fastai.text import load_learner, AUROC
from sklearn.metrics import f1_score
import torch
torch.device('cpu')
import os 

flask_app = Flask(__name__)
app = Api(app = flask_app,
          version = "1.0",
          title = "Name Recorder",
          description = "Manage names of various users of the application")

name_space = app.namespace('main', description='Manage names')
body_require = app.model('text',
                  {'text': fields.String(required = True, description="input text", help="Cannot be blank.",
                  example="5 mã trắng cửa bán rồi thì FLC tí nữa thôi là lại tím lịm")})
#load model
# @np_func
def f1(inp,targ): return f1_score(targ, np.argmax(inp, axis=-1))
model_dir = os.getcwd()
learn = load_learner(model_dir, file='stock_sentiment_model.pkl')
learn.to_fp32() # using with cpu
#
@name_space.route("/")
class MainClass(Resource):
  @app.doc(responses={ 200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error' })
  @app.expect(body_require)
  def post(self):
    text = request.json['text']
    predicted_value = learn.predict(text.lower())[2][1].item()
    return jsonify(predicted_value)

# Run terminal: FLASK_APP=app.py flask run
if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', debug=True, use_reloader=False)  # `use_reloader=False` = flask not reload twice, but not reload when changes


