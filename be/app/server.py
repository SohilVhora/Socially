import torch
import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
)
from torch.nn.functional import softmax

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Deferred Model Loading ======
asr_processor = None
asr_model = None
emotion_model = None
id2label = None
llm_tokenizer = None
llm_model = None


def init_models() -> None:
    """Load heavy models on first use."""
    global asr_processor, asr_model, emotion_model, id2label, llm_tokenizer, llm_model
    if asr_model is None:
        asr_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        asr_model = (
            Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            .to(device)
            .eval()
        )

        emotion_model = (
            Wav2Vec2ForSequenceClassification.from_pretrained(
                "app/models/emotion_model", local_files_only=True
            )
            .to(device)
            .eval()
        )
        id2label = emotion_model.config.id2label

        token = "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"
        llm_tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-llm-1.3b-base",
            use_auth_token=token,
            trust_remote_code=True,
        )
        llm_model = (
            AutoModelForCausalLM.from_pretrained(
                "deepseek-ai/deepseek-llm-1.3b-base",
                use_auth_token=token,
                device_map="auto",
                trust_remote_code=True,
            )
            .eval()
        )
        print("Models initialized and ready.")

# ====== Initialize VAD ======
vad = webrtcvad.Vad(2)  # Aggressiveness: 0 (least) to 3 (most)
sample_rate = 16000
frame_duration = 30  # ms
frame_size = int(sample_rate * frame_duration / 1000) * 2  # 16-bit = 2 bytes
min_speech_frames = 5
silence_timeout = 20


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    init_models()
    buffer = b""
    speech_started = False
    silence_counter = 0

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()

            # We might get multiple frames at once; process all valid VAD frames
            for i in range(0, len(audio_chunk), frame_size):
                frame = audio_chunk[i:i+frame_size]
                if len(frame) < frame_size:
                    continue

                is_speech = vad.is_speech(frame, sample_rate)

                if is_speech:
                    buffer += frame
                    speech_started = True
                    silence_counter = 0
                elif speech_started:
                    silence_counter += 1
                    if silence_counter > silence_timeout and len(buffer) > 0:
                        # Finalize utterance
                        audio_input = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0

                        # ASR
                        inputs = asr_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
                        with torch.no_grad():
                            logits = asr_model(inputs.input_values.to(device)).logits
                            pred_ids = torch.argmax(logits, dim=-1)
                            transcription = asr_processor.batch_decode(pred_ids)[0].strip()

                        # Emotion
                        emo_input_tensor = torch.tensor(audio_input).unsqueeze(0).to(device)
                        with torch.no_grad():
                            emo_logits = emotion_model(emo_input_tensor).logits
                        probs = softmax(emo_logits, dim=1)
                        emotion = id2label[torch.argmax(probs, dim=1).item()]
                        print({"text": transcription, "emotion": emotion})

                        await websocket.send_json({"text": transcription, "emotion": emotion})

                        # Reset state
                        buffer = b""
                        speech_started = False
                        silence_counter = 0

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        try:
            if websocket.client_state.name == "CONNECTED":
                await websocket.send_json({"error": str(e)})
        except:
            pass
        print("Error:", e)
