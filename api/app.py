# uvicorn app:app --reload
# ngrok http 8000
import base64
import cv2
import time
import os
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from YOLOv8 import YOLOv8
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

loaded_model = YOLOv8()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image_base64: str


# My endpoint
@app.get("/")
async def read_root():
    return {"blind guide api"}


@app.post("/", tags=["list"])
async def post_detect_image(image_request: ImageRequest):
    encoded_image = image_request.image_base64
    img_data = base64.b64decode(encoded_image)
    results = classify_image(img_data)
    return results


@app.websocket("/detect-image/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("the connection accepted")
    cnt = 0
    while True:
        try:
            encoded_image = await websocket.receive_text()
            decoded_image = base64.b64decode(encoded_image)
            results = classify_image(decoded_image)
            print(results)
            print("counting", cnt)
            cnt += 1
            await websocket.send_json(results)
            print("sent")
        except:
            pass
            break


def classify_image(img_data):
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    conf = 0.7
    txt, image, boxes = loaded_model.predict(img, conf)
    return {"list": txt, "boxes": boxes}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
