import torch
import torch.nn as nn
import numpy as np
import eng_to_ipa

import interfaces


def getASRModel() -> nn.Module:
    model, decoder, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_stt",
        language="en",
        device=torch.device("cpu"),
    )

    return (model, decoder)


class NeuralASR(interfaces.IASRModel):
    word_locations_in_samples = None
    audio_transcript = None

    def __init__(self, model: torch.nn.Module, decoder) -> None:
        super().__init__()
        self.model = model
        self.decoder = decoder  # Decoder from CTC-outputs to transcripts

    def getTranscript(self) -> str:
        """Get the transcripts of the process audio"""

        return self.audio_transcript

    def getWordLocations(self) -> list:
        """Get the pair of words location from audio"""

        return self.word_locations_in_samples

    def processAudio(self, audio: torch.Tensor):
        """Process the audio"""
        audio_length_in_samples = audio.shape[1]
        with torch.inference_mode():
            nn_output = self.model(audio)

            self.audio_transcript, self.word_locations_in_samples = self.decoder(
                nn_output[0, :, :].detach(), audio_length_in_samples, word_align=True
            )


class EngPhonemConverter(interfaces.ITextToPhonemModel):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def convertToPhonem(self, sentence: str) -> str:
        phonem_representation = eng_to_ipa.convert(sentence)
        phonem_representation = phonem_representation.replace("*", "")
        return phonem_representation
