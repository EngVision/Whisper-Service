import os
from tempfile import NamedTemporaryFile
import requests
from torchaudio.transforms import Resample

import torch
import numpy as np
import audioread
import json

import utils
import trainer


trainer_SST_lambda = {}
trainer_SST_lambda["en"] = trainer.getTrainer()

transform = Resample(orig_freq=48000, new_freq=16000)


def lambda_handler(event, context):
    data = json.loads(event["body"])
    
    original = data["original"]
    response = requests.get(f"{os.environ.get('BUCKET_URL')}{data['fileId']}")
    temp = NamedTemporaryFile(delete=True)
    temp.write(response.content)

    signal, fs = audioread_load(temp.name)

    signal = transform(torch.Tensor(signal)).unsqueeze(0)

    result = trainer_SST_lambda["en"].processAudioForGivenText(signal, original)

    temp.close()

    original_ipa_transcript = " ".join(
        [word[0] for word in result["real_and_transcribed_words_ipa"]]
    )

    original_transcript = " ".join(
        [word[0] for word in result["real_and_transcribed_words"]]
    )

    words_real = original_ipa_transcript.lower().split()
    mapped_words = result["recording_ipa"].split()

    correct_letters = ""
    for idx, word_real in enumerate(words_real):
        mapped_letters, mapped_letters_indices = utils.get_best_mapped_words(
            mapped_words[idx], word_real
        )

        is_letter_correct = utils.getWhichLettersWereTranscribedCorrectly(
            word_real, mapped_letters
        )

        correct_letters += (
            "".join([str(is_correct) for is_correct in is_letter_correct]) + " "
        )

    res = {
        "_id": data['fileId'],
        "file_id": data['fileId'],
        "original_transcript": original_transcript,
        "voice_transcript": result["recording_transcript"],
        "original_ipa_transcript": original_ipa_transcript,
        "voice_ipa_transcript": result["recording_ipa"],
        "pronunciation_accuracy": str(int(result["pronunciation_accuracy"])),
        "correct_letters": correct_letters,
    }

    return res


# From Librosa
def audioread_load(path, offset=0.0, duration=None, dtype=np.float32):
    """Load an audio buffer using audioread.

    This loads one block at a time, and then concatenates the results.
    """

    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[: s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev) :]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    return y, sr_native


# From Librosa
def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)
