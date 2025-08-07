import os
import asyncio
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
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# ====== Lifespan Management for Model Loading ======

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load environment variables and ML models before the app starts.
    """
    print("Loading environment variables...")
    # This path is robust and finds the .env file in the `be` directory
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    
    print("Loading models...")
    init_models()
    print("Models loaded successfully.")
    yield
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
    
    asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

    emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained("app/models/emotion_model", local_files_only=True).to(device).eval()
    id2label = emotion_model.config.id2label

    token = os.getenv("HUGGING_FACE_TOKEN")
    if not token:
        raise ValueError("HUGGING_FACE_TOKEN environment variable not set! Please check your .env file.")

    llm_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-1.3b-base", use_auth_token=token, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-1.3b-base", use_auth_token=token, device_map="auto", trust_remote_code=True).eval()

def generate_llm_response(text: str, emotion: str) -> str:
    """
    Generates a conversational response from the LLM.
    """
    # This prompt helps the AI understand the context and its persona.
    prompt = f"The user, who seems to be feeling {emotion}, just said: '{text}'. Respond to them in a friendly and brief conversational manner."
    
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Adjust max_new_tokens to control the length of the AI's response.
        output_ids = llm_model.generate(**inputs, max_new_tokens=60, num_return_sequences=1, pad_token_id=llm_tokenizer.eos_token_id)
    
    response = llm_tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def transcribe_audio(audio_input: np.ndarray) -> str:
    """Run speech-to-text transcription."""
    inputs = asr_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = asr_model(inputs.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
    return asr_processor.batch_decode(pred_ids)[0].strip()


def analyze_emotion_audio(audio_input: np.ndarray) -> str:
    """Detect the speaker's emotion."""
    emo_input_tensor = torch.tensor(audio_input).unsqueeze(0).to(device)
    with torch.no_grad():
        emo_logits = emotion_model(emo_input_tensor).logits
    probs = softmax(emo_logits, dim=1)
    return id2label[torch.argmax(probs, dim=1).item()]


async def process_audio_chunk(websocket: WebSocket, audio_input: np.ndarray) -> None:
    """Handle ASR, emotion analysis and LLM generation in background threads."""
    transcription_task = asyncio.to_thread(transcribe_audio, audio_input)
    emotion_task = asyncio.to_thread(analyze_emotion_audio, audio_input)
    transcription, emotion = await asyncio.gather(transcription_task, emotion_task)

    await websocket.send_json({"text": transcription, "emotion": emotion})

    ai_response = await asyncio.to_thread(generate_llm_response, transcription, emotion)
    print(f"User (emotion: {emotion}): {transcription}")
    print(f"AI Response: {ai_response}")
    await websocket.send_json({"ai_response": ai_response})

# ====== VAD Configuration ======
vad = webrtcvad.Vad(2)
sample_rate = 16000
frame_duration = 30  # ms
frame_size = int(sample_rate * frame_duration / 1000) * 2
silence_timeout_frames = 20

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    buffer = b""
    speech_started = False
    silence_counter = 0

    try:
        while True:
            audio_chunk = await websocket.receive_bytes()

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
                    if silence_counter > silence_timeout_frames and len(buffer) > 0:
                        audio_input = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0

                        asyncio.create_task(process_audio_chunk(websocket, audio_input))

                        # Reset buffer for the next utterance
                        buffer = b""
                        speech_started = False
                        silence_counter = 0

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        try:
            if websocket.client_state.name == "CONNECTED":
                await websocket.send_json({"error": error_message})
        except Exception as e_send:
            print(f"Failed to send error to client: {e_send}")
