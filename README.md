# LangChainSummarizer

LangChainSummarizer is a simple web application that provides an AI-powered text summarization service 
using FastAPI and LangChain. 
Users can submit text input, and the application returns a concise summary of the text. 
The summarization is powered by a pre-trained model from Hugging Face's Transformers library, 
ensuring high-quality and accurate summaries. 
## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   On Windows use `venv\Scripts\activate`
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. Open your browser and go to http://127.0.0.1:8000/ to access the API and test the endpoint.


### Endpoint

- POST /summarize

Request Body:

```json
{
  "text": "Your text to be summarized here"
}
```

Response:

```json
{
  "summary": "Summarized text here"
}
```

By following these steps, you'll have a working FastAPI application that uses LangChain and a pre-trained Hugging Face model to summarize text.
