from fastapi import FastAPI,File,Request, UploadFile
from fastapi.responses import JSONResponse,HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app=FastAPI()

model=tf.keras.models.load_model("mlp_model.keras")
class_names=['Flower','Bird','Human','Elephant','Car']
image_size=(64,64)

templates=Jinja2Templates(directory='templates')




@app.get('/',response_class=HTMLResponse)
async def home(request:Request):
    print("Serving index.html")
    return templates.TemplateResponse('index.html',{'request':request})

@app.post("/predict/")
async def predict(file:UploadFile=File(...)):
    try:
        contents=await file.read()
        img=Image.open(io.BytesIO(contents)).convert("RGB")
        img=img.resize(image_size)
        img_array=np.array(img,dtype=np.float32)
        img_array=np.expand_dims(img_array,axis=0)

        predictions=model.predict(img_array)
        predicted_class=class_names[np.argmax(predictions)]
        confidence=float(np.max(predictions))

        return JSONResponse({
            'class':predicted_class,
            'confidence':confidence
        })
    except Exception as e:
        return JSONResponse(status_code=500,content={'error':str(e)})

if __name__=='__main__':
    uvicorn.run('main:app',host='127.0.0.1',port=8802,reload=True)

