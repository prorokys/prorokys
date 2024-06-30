from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
from fastapi.responses import HTMLResponse
from langchain.docstore.document import Document
import uvicorn


app = FastAPI()
# Initialize FastAPI

templates = Jinja2Templates(directory="templates")
# Initialize Jinja2Templates


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Render index.html
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/summarize", response_class=HTMLResponse)
def summarize(request: Request, text: str = Form(...)):
    # Load model from HuggingFace
    model = HuggingFacePipeline.from_model_id(
        model_id="sshleifer/distilbart-cnn-12-6",
        task="summarization",
        model_kwargs={"max_length": 1000},
    )
    # Load text to summarize
    data = [Document(page_content=text)]
    # Load summarize chain from model
    chain = load_summarize_chain(model)
    # Run summarize chain on text to get summary
    summary = chain.run(data)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "summary": summary, "original_text": text},
    )


if __name__ == "__main__":
    # Run FastAPI server
    uvicorn.run(app, host="127.0.0.1", port=8001)
