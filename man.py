from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import uvicorn


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.post("/summarize", response_class=HTMLResponse)
def summarize(request: Request, text: str = Form(...)):

    model = HuggingFacePipeline.from_model_id(
        model_id="sshleifer/distilbart-cnn-12-6",
        task="summarization",
        model_kwargs={"max_length": 1000},
    )

    data = [Document(page_content=text)]
    chain = load_summarize_chain(model)

    summary = chain.run(data)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "summary": summary, "original_text": text},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
