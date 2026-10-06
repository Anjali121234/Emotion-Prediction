import pickle
import re
from contextlib import asynccontextmanager
import numpy as np
from fastapi.responses import FileResponse

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



model_path="Artifacts/BiGRU_Modle.keras"
tokenizer_path="Artifacts/tokenizer.pkl"
max_sequence_length=50
emotion_labels=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

def preprocess_text(text: str)->str:
    text = text.lower()
    text = re.sub(r"'","",text)
    text = re.sub(r"[^a-z0-9\s]"," ", text)
    text = re.sub(r"\s+", " ",text).strip()
    return text

class TextInput(BaseModel):
    text:str = Field(...,min_length=1,max_length=2000,
                     description="give a sentence",
                     json_schema_extra={"example":"I like myself"})

class PredictionResponse(BaseModel):
    text:str
    predicted_emotion:str
    confidence:float
    all_probabilities:dict[str,float]

class HealthResponse(BaseModel):
    status:str
    model_loaded:bool

dl_model={}

@asynccontextmanager
async def lifespan(api: FastAPI):
    print('loading the model and tokenizer...')
    dl_model["BiGRU"]=load_model(model_path)
    with open(tokenizer_path,'rb') as file:
        dl_model["Tokenizer"]=pickle.load(file)
    print("model are loaded successfully...")

    yield 

    dl_model.clear()


app = FastAPI(title="Sentiment Analysis API",lifespan=lifespan)


# 5. Mount the static files to the FastAPI app
# A. Enable CORS (Cross-Origin Resource Sharing) to allow requests from different origins.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount('/static',StaticFiles(directory="static"),name="static")


#  API Endpoints.

@app.get('/',include_in_schema=False)
def server_ui():
    return FileResponse('static/index.html')


@app.get('/health',response_model=HealthResponse)
def health_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))

@app.post('/predict',response_model=PredictionResponse)
def predict_emotion(text_input:TextInput):
    BiGRU_model=dl_model.get("BiGRU")
    tokenizer_model=dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503,detail="Model is not loaded, try again")

    cleaned_text=preprocess_text(text_input.text)
    tokenized_text=tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )
    print(cleaned_text)

    probabilities= BiGRU_model.predict(padded_sequence)[0]

    top_emotion_index=int(np.argmax(probabilities))
    all_probabilities_dict =  {
        label: float(prob) for prob, label in zip(probabilities, emotion_labels)
    }

    return PredictionResponse(
        text = text_input.text,
        predicted_emotion = emotion_labels[top_emotion_index],
        confidence = float(probabilities[top_emotion_index]), 
        all_probabilities = all_probabilities_dict
    )
    


