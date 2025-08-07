import numpy as np
import torch
import webrtcvad
from torch.nn.functional import softmax

def load_models(asr_model_path, emotion_model_path, device):
    from transformers import AutoProcessor, AutoModelForCTC, AutoTokenizer, AutoModelForSequenceClassification

    asr_model_name = "facebook/wav2vec2-base-960h"
    asr_processor = AutoProcessor.from_pretrained(asr_model_name)
    asr_model = AutoModelForCTC.from_pretrained(asr_model_name).to(device)

    emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_path, local_files_only=True)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(
        emotion_model_path, local_files_only=True
    ).to(device)
    id2label = emotion_model.config.id2label

    return asr_processor, asr_model, emotion_tokenizer, emotion_model, id2label

def process_audio_chunk(buffer, asr_processor, asr_model, emotion_tokenizer, emotion_model, id2label, device, sample_rate=16000):
    # Convert buffer to waveform
    audio = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
    audio = torch.tensor(audio).unsqueeze(0).to(device)

    # ASR
    inputs = asr_processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = asr_model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.batch_decode(pred_ids)[0].strip()

    # Emotion
    emo_inputs = emotion_tokenizer(transcription, return_tensors="pt", padding=True, truncation=True)
    emo_inputs = {k: v.to(device) for k, v in emo_inputs.items()}
    with torch.no_grad():
        emo_logits = emotion_model(**emo_inputs).logits
    probs = softmax(emo_logits, dim=1)
    emo_id = torch.argmax(probs, dim=1).item()
    emotion = id2label[emo_id]

    return transcription, emotion
