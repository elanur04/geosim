from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read the file to determine its size
    content = await file.read()
    file_size = len(content)
    
    return {
        "filename": file.filename,
        "size": file_size
    }
