from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("OPENAI_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def get_models():
    r = requests.get("https://api.openai.com/v1/models", headers=HEADERS)
    return r.json().get("data", []) if r.ok else []


def get_fine_tunes():
    r = requests.get("https://api.openai.com/v1/fine_tunes", headers=HEADERS)
    return r.json().get("data", []) if r.ok else []


def get_fine_tune_info(ft_id):
    r = requests.get(f"https://api.openai.com/v1/fine_tunes/{ft_id}", headers=HEADERS)
    return r.json() if r.ok else {"error": r.text}


@app.route('/')
def index():
    models = get_models()
    fine_tunes = get_fine_tunes()

    fine_tune_map = {ft['fine_tuned_model']: ft['id'] for ft in fine_tunes if 'fine_tuned_model' in ft}
    openai_models = [m['id'] for m in models if not m['id'].startswith("ft:")]
    fine_tuned_models = [m['id'] for m in models if m['id'].startswith("ft:")]

    return render_template("index.html",
                           openai_models=openai_models,
                           fine_tuned_models=fine_tuned_models,
                           fine_tune_map=fine_tune_map)


@app.route('/model/<model_id>')
def model_info(model_id):
    ftid = request.args.get("ftid")
    if ftid:
        info = get_fine_tune_info(ftid)
    else:
        r = requests.get(f"https://api.openai.com/v1/models/{model_id}", headers=HEADERS)
        info = r.json() if r.ok else {"error": r.text}
    return render_template("model_info.html", model_id=model_id, info=info)


@app.route('/api/models')
def api_models():
    models = get_models()
    return jsonify(models)


@app.route('/api/fine_tunes')
def api_fine_tunes():
    fine_tunes = get_fine_tunes()
    return jsonify(fine_tunes)


@app.route('/api/model/<model_id>')
def api_model_info(model_id):
    ftid = request.args.get("ftid")
    if ftid:
        info = get_fine_tune_info(ftid)
    else:
        r = requests.get(f"https://api.openai.com/v1/models/{model_id}", headers=HEADERS)
        info = r.json() if r.ok else {"error": r.text}
    return jsonify(info)


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
