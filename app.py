from flask import Flask, jsonify, request
from flask_cors import CORS
from tempfile import NamedTemporaryFile
import whisper
import threading
import pymongo
import json
import requests
from enums import StatusEnum

import speech_evaluation


model = whisper.load_model("base.en")
app = Flask(__name__)
CORS(app)

mongo_client = pymongo.MongoClient(
    "mongodb+srv://admin:5LnPkKh2dx1JFEI6@cluster0.sugklve.mongodb.net/dev?retryWrites=true&w=majority")
mongo_db = mongo_client["dev"]
speech_to_text_collection = mongo_db["speechtotext"]
speech_evaluation_collection = mongo_db["speechevaluation"]


def transcribe_background(file_id):
    response = requests.get(f'https://dev-bucket.engvision.edu.vn/{file_id}')
    temp = NamedTemporaryFile(delete=True)
    temp.write(response.content)
    result = model.transcribe(temp.name)
    speech_to_text_collection.update_one({"_id": file_id},
                                         {"$set": {"text": result['text'],
                                                   "status": StatusEnum.COMPLETED.value}})
    temp.close()


@app.route("/")
def root_handler():
    return "Hello from EngVision Whisper!"


@app.route("/stt/<file_id>", methods=["POST"])
def whisper_handler(file_id):
    result = speech_to_text_collection.find_one({"_id": file_id})
    if result is not None:
        return jsonify(result)

    response = requests.get(f'https://dev-bucket.engvision.edu.vn/{file_id}')
    if response.status_code != 200:
        return jsonify({"error": "File not found"}), 404

    data = {
        "_id": file_id,
        "file_id": file_id,
        "text": None,
        "status": StatusEnum.PROCESSING.value}
    speech_to_text_collection.insert_one(data)
    threading.Thread(
        target=transcribe_background, args=(file_id,)
    ).start()

    return jsonify(data)


@app.route("/stt/<file_id>", methods=["GET"])
def check_status(file_id):
    result = speech_to_text_collection.find_one({"_id": file_id})
    if result is None:
        return jsonify({"error": "File not found"}), 404

    return jsonify(result)


@app.route("/speech-evaluation", methods=["POST"])
def handle_speech_evaluation():
    event = {"body": json.dumps(request.get_json(force=True))}
    lambda_correct_output = speech_evaluation.lambda_handler(event, [])
    speech_evaluation_collection.insert_one(lambda_correct_output)
    return jsonify(lambda_correct_output)


if __name__ == "__main__":
    app.run(debug=True)
