# uvicorn app:app --port 12000 --reload
# ngrok http 12000
import base64
import cv2
import io
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from YOLOv8 import YOLOv8

app = FastAPI()

loaded_model = YOLOv8()


class ImageRequest(BaseModel):
    image_base64: str


# My endpoint
@app.get("/")
async def read_root():
    return {"blind guide api"}


@app.post("/detect-image/", tags=["list"])
async def post_detect_image(image_request: ImageRequest):
    encoded_image = image_request.image_base64
    img_data = base64.b64decode(encoded_image)
    results = classify_image(img_data)
    return results


@app.websocket("/detect-image/")
async def websocket_endpoint(websocket: WebSocket):
    while True:
        encoded_image = await websocket.receive_text()
        img_data = base64.b64decode(encoded_image)
        results = classify_image(img_data)
        await websocket.send_json(results)


def classify_image(img_data):
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    conf = 0.7
    txt, image = loaded_model.predict(img, conf)
    retval, buffer = cv2.imencode(".jpg", image)
    encoded_processed_image = base64.b64encode(buffer)
    return {"list": txt, "image": encoded_processed_image}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=12000)
