import torch
import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
)
from torch.nn.functional import softmax
from contextlib import asynccontextmanager

# ====== Lifespan Management for Model Loading ======

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the ML models before the application starts accepting requests.
    """
    print("Loading models...")
    init_models()
    print("Models loaded successfully.")
    yield
    # Add any cleanup logic here if needed, e.g., releasing GPU memory
    print("Application shutting down.")

app = FastAPI(lifespan=lifespan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Model Placeholders ======
asr_processor = None
asr_model = None
emotion_model = None
id2label = None
llm_tokenizer = None
llm_model = None

def init_models() -> None:
    """
    Initializes and loads all the machine learning models into global variables.
    """
    global asr_processor, asr_model, emotion_model, id2label, llm_tokenizer, llm_model
    
    # Load ASR Model
    asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = (
        Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        .to(device)
        .eval()
    )

    # Load Emotion Classification Model
    emotion_model = (
        Wav2Vec2ForSequenceClassification.from_pretrained(
            "app/models/emotion_model", local_files_only=True
        )
        .to(device)
        .eval()
    )
    id2label = emotion_model.config.id2label

    # Load Large Language Model (LLM)
    token = "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" # Note: Consider moving this to an environment variable
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

# ====== VAD Configuration ======
vad = webrtcvad.Vad(2)  # Aggressiveness: 0 (least) to 3 (most)
sample_rate = 16000
frame_duration = 30  # ms
frame_size = int(sample_rate * frame_duration / 1000) * 2  # Each sample is 16-bit (2 bytes)
silence_timeout_frames = 20 # Number of silent frames to wait before processing

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    buffer = b""
    speech_started = False
    silence_counter = 0

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()

            # Process audio chunk in VAD-sized frames
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
                    # Frame is silence, but we were just speaking
                    silence_counter += 1
                    if silence_counter > silence_timeout_frames and len(buffer) > 0:
                        # Process the audio buffer after a period of silence
                        audio_input = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0

                        # ASR (Speech-to-Text)
                        inputs = asr_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
                        with torch.no_grad():
                            logits = asr_model(inputs.input_values.to(device)).logits
                            pred_ids = torch.argmax(logits, dim=-1)
                            transcription = asr_processor.batch_decode(pred_ids)[0].strip()

                        # Emotion Detection
                        emo_input_tensor = torch.tensor(audio_input).unsqueeze(0).to(device)
                        with torch.no_grad():
                            emo_logits = emotion_model(emo_input_tensor).logits
                        probs = softmax(emo_logits, dim=1)
                        emotion = id2label[torch.argmax(probs, dim=1).item()]
                        
                        print({"text": transcription, "emotion": emotion})
                        await websocket.send_json({"text": transcription, "emotion": emotion})

                        # Reset for next utterance
                        buffer = b""
                        speech_started = False
                        silence_counter = 0

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        try:
            # Try to send an error message to the client before closing
            if websocket.client_state.name == "CONNECTED":
                await websocket.send_json({"error": error_message})
        except Exception as e_send:
            print(f"Failed to send error to client: {e_send}")
