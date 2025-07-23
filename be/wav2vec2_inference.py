import soundfile as sf
import torch
import numpy as np
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor

class Wave2Vec2Inference:
    def __init__(self, model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.use_lm_if_possible = use_lm_if_possible
        self.hotwords = hotwords

        # Load processor
        if use_lm_if_possible:
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        # Load model
        self.model = AutoModelForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return "", 0.0

        inputs = self.processor(
            torch.tensor(audio_buffer),
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            if attention_mask is not None:
                logits = self.model(input_values=input_values, attention_mask=attention_mask).logits
            else:
                logits = self.model(input_values=input_values).logits

        if hasattr(self.processor, 'decode') and self.use_lm_if_possible:
            transcription_obj = self.processor.decode(
                logits[0].cpu().numpy(),
                hotwords=self.hotwords,
                output_word_offsets=True
            )
            confidence = transcription_obj.lm_score / max(1, len(transcription_obj.text.split(" ")))
            transcription = transcription_obj.text
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            confidence = self.confidence_score(logits, predicted_ids)

        return transcription, confidence


    def confidence_score(self, logits, predicted_ids):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_scores = probs.gather(-1, predicted_ids.unsqueeze(-1)).squeeze(-1)

        mask = torch.logical_and(
            predicted_ids != self.processor.tokenizer.pad_token_id,
            predicted_ids != self.processor.tokenizer.word_delimiter_token_id
        )

        character_scores = pred_scores.masked_select(mask)
        return character_scores.mean().item()

    def file_to_text(self, filename):
        audio_input, samplerate = sf.read(filename)
        assert samplerate == 16000, "Sample rate must be 16kHz"
        return self.buffer_to_text(audio_input)

if __name__ == "__main__":
    print("Running Wav2Vec2 ASR Inference...")
    asr = Wave2Vec2Inference("facebook/wav2vec2-base-960h")
    transcription, confidence = asr.file_to_text("test.wav")
    print("Transcription:", transcription)
    print("Confidence:", confidence)
