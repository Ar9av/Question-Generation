import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))

import numpy as np
import tensorflow as tf
from flask import Flask, current_app, redirect, request

import flags
from helpers import preprocessing
from instance import AQInstance

from dockp.utils.api_utils import APIHelper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
FLAGS = tf.app.flags.FLAGS

mem_limit=0.9
# model_slug_list = ['MALUUBA-CROP-SET-GLOVE/1535568489','SEQ2SEQ/1533280948','qgen-s2s-filt1']
model_slug_list = ['RL-S2S-1544356761']
model_slug_curr = model_slug_list[0]

models_base_dir = os.path.join(os.path.dirname(__file__), "../../models")
chkpt_path = os.path.join(models_base_dir, "qgen", model_slug_curr)
with open(chkpt_path+'/vocab.json') as f:
    vocab = json.load(f)

GENERATOR = AQInstance(vocab=vocab)
GENERATOR.load_from_chkpt(chkpt_path)

app = Flask(__name__)

@app.route("/")
def index():
    return redirect("/static/demo.htm")

@app.route("/generate_from_pdf")
def generate_from_pdf():

    kp_id = request.args["kp_id"]
    doc_id = request.args["doc_id"]
    options = request.args["options"]
    # options = {"api_server": "http://127.0.0.1:3000"}
    kp_stems = APIHelper.fetch_kp_stems(kp_id, options)
    chunks = APIHelper.fetch_chunks(kp_stems, options)
    return chunks

@app.route("/api/generate")
def get_q():

    ctxt = request.args['context']
    ans = request.args['answer']
    ans_pos = int(request.args.get('ans_pos', ctxt.find(ans)))

    if FLAGS.filter_window_size_before >-1:
        ctxt,ans_pos = preprocessing.filter_context(ctxt, ans_pos, FLAGS.filter_window_size_before, FLAGS.filter_window_size_after, FLAGS.filter_max_tokens)
    if ans_pos > -1:
        q =GENERATOR.get_q(ctxt.encode(), ans.encode(), ans_pos)
        return q
    else:
        return "Couldnt find ans in context!"

@app.route("/api/generate_batch", methods=['POST'])
def get_q_batch():
    args = request.get_json()

    ctxts, ans, ans_pos = map(list, zip(*args['queries']))

    if FLAGS.filter_window_size_before >-1:
        for i in range(len(ctxts)):
            ctxts[i], ans_pos[i] = preprocessing.filter_context(ctxts[i], ans_pos[i], FLAGS.filter_window_size_before, FLAGS.filter_window_size_after, FLAGS.filter_max_tokens)
            
            ctxts[i] = ctxts[i].encode()
            ans[i] = ans[i].encode()

    
    if len(ctxts) > 32:
        qs = []
        for b in range(len(ctxts)//32 + 1):
            start_ix = b * 32
            end_ix = min(len(ctxts), (b + 1) * 32)
            qs.extend(GENERATOR.get_q_batch(ctxts[start_ix:end_ix], ans[start_ix:end_ix], ans_pos[start_ix:end_ix]))
    else:
        qs = GENERATOR.get_q_batch(ctxts, ans, ans_pos)

    resp = {'status': 'OK',
            'results': [{'q': qs[i], 'a': ans[i].decode()} for i in range(len(qs))]}
    return json.dumps(resp)
    

@app.route("/api/ping")
def ping():
    return GENERATOR.ping()

@app.route("/api/model_list")
def model_slug():
    return json.dumps(model_list)

@app.route("/api/model_current")
def model_list():
    return model_slug_curr

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5004, processes=1)
