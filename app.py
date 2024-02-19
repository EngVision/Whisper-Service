from flask import Flask, jsonify, request, abort, url_for
from flask_cors import CORS
from tempfile import NamedTemporaryFile
import whisper
import threading
import uuid
import json

import speech_evaluation


model = whisper.load_model("base")
app = Flask(__name__)
CORS(app)

tasks = {}


def transcribe_background(file_id, file_path):
    result = model.transcribe(file_path)
    tasks[file_id] = result["text"]


@app.route("/")
def root_handler():
    return "Hello from EngVision Whisper!"


@app.route("/stt", methods=["POST"])
def whisper_handler():
    if not request.files or len(request.files) != 1:
        abort(400)

    file_id = str(uuid.uuid4())
    for _, file in request.files.items():
        temp = NamedTemporaryFile(delete=False)
        file.save(temp.name)
        tasks[file_id] = "processing"
        threading.Thread(
            target=transcribe_background, args=(file_id, temp.name)
        ).start()

    return jsonify(
        {
            "status": "processing",
            "check_url": url_for("check_status", file_id=file_id, _external=True),
        }
    )


@app.route("/stt/<file_id>", methods=["GET"])
def check_status(file_id):
    if file_id not in tasks:
        return jsonify({"error": "File not found"}), 404

    status = tasks[file_id]
    if status == "processing":
        return jsonify({"status": "processing"})
    else:
        return jsonify({"status": "completed", "result": status})


@app.route("/speech-evaluation", methods=["POST"])
def handle_speech_evaluation():
    event = {"body": json.dumps(request.get_json(force=True))}
    lambda_correct_output = speech_evaluation.lambda_handler(event, [])
    return lambda_correct_output


if __name__ == "__main__":
    app.run(debug=True)
