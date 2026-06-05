from fastapi import FastAPI, UploadFile, File
import shutil
from app import process_pdf, ask_question

app = FastAPI()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process_pdf(path)

    return {"message": "PDF processed"}


@app.get("/ask")
def ask(q: str):

    answer = ask_question(q)

    return {"answer": answer}