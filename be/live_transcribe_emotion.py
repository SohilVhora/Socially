import sounddevice as sd
import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Wav2Vec2ForCTC
import time

# === Paths ===
EMOTION_MODEL_PATH = "models/asr_model"
ASR_MODEL_NAME = "facebook/wav2vec2-base-960h"

# === Load models ===
emotion_processor = Wav2Vec2Processor.from_pretrained(EMOTION_MODEL_PATH)
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH).eval()

asr_processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)
asr_model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_NAME).eval()

# === Labels ===
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# === Mic settings ===
MIC_INDEX = 4            # your working mic device
MIC_SAMPLE_RATE = 48000  # your hardware rate
MODEL_SAMPLE_RATE = 16000
DURATION = 4             # seconds

resampler = Resample(orig_freq=MIC_SAMPLE_RATE, new_freq=MODEL_SAMPLE_RATE)

def analyze_audio(audio_np):
    # Convert mic input to tensor and resample
    audio_tensor = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)  # [1, T]
    audio_resampled = resampler(audio_tensor).squeeze(0).numpy()             # [T]

    # Ensure it's always a 1D float32 numpy array
    if not isinstance(audio_resampled, np.ndarray):
        audio_resampled = np.array([audio_resampled], dtype=np.float32)
    elif audio_resampled.ndim == 0:
        audio_resampled = np.expand_dims(audio_resampled, axis=0)
    audio_resampled = audio_resampled.astype(np.float32)

    # Skip if audio too short or silent
    if len(audio_resampled) < 1000 or np.allclose(audio_resampled, 0, atol=1e-3):
        return "no speech", "silence", [0.0] * len(emotion_labels)

    # === Emotion Prediction ===
    emo_inputs = emotion_processor(
        [audio_resampled],  # make it a batch of 1
        sampling_rate=MODEL_SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        emo_logits = emotion_model(**emo_inputs).logits
        emo_probs = F.softmax(emo_logits, dim=1)
        emo_pred = torch.argmax(emo_probs, dim=1).item()
        emotion = emotion_labels[emo_pred]

    # === ASR Prediction ===
    asr_inputs = asr_processor(
        [audio_resampled],  # make it a batch of 1
        sampling_rate=MODEL_SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        asr_logits = asr_model(**asr_inputs).logits
        asr_pred_ids = torch.argmax(asr_logits, dim=-1)
        transcription = asr_processor.batch_decode(asr_pred_ids)[0]

    return transcription.strip(), emotion, emo_probs.squeeze().tolist()



def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️ Mic warning:", status)
    audio_np = indata[:, 0].copy()
    print("\n🔊 Processing mic input...")
    text, emotion, probs = analyze_audio(audio_np)
    print(f"📝 Text: {text}")
    print(f"🎭 Emotion: {emotion}")
    print(f"📊 Confidence: {dict(zip(emotion_labels, [round(p, 2) for p in probs]))}")

# === Start Stream ===
print(f"🎙️ Listening on device {MIC_INDEX} at {MIC_SAMPLE_RATE}Hz...")

sd.default.device = (MIC_INDEX, None)

try:
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=MIC_SAMPLE_RATE,
                        blocksize=int(MIC_SAMPLE_RATE * DURATION)):
        while True:
            time.sleep(DURATION)
except KeyboardInterrupt:
    print("\n🛑 Stopped.")
