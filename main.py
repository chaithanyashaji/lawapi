import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import warnings

# Logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Starting FastAPI app...")

# Suppress warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()
TOGETHER_AI_API = os.getenv("TOGETHER_AI")
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", "./cache")

# Set cache directory for transformers
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE

# Validate environment variables
if not TOGETHER_AI_API:
    raise ValueError("Environment variable TOGETHER_AI_API is missing. Please set it in your .env file.")

# Ensure cache directory exists
if not os.path.exists(TRANSFORMERS_CACHE):
    os.makedirs(TRANSFORMERS_CACHE)

# Initialize embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"},
)

# Ensure FAISS vectorstore is loaded properly
try:
    db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2, "max_length": 512})
except Exception as e:
    logger.error(f"Error loading FAISS vectorstore: {e}")
    raise RuntimeError("FAISS vectorstore could not be loaded. Ensure the vector database exists.")

# Define the prompt template
prompt_template = """<s>[INST]As a legal chatbot specializing in the Indian Penal Code, provide a concise and accurate answer based on the given context. Avoid unnecessary details or unrelated content. Only respond if the answer can be derived from the provided context; otherwise, say "The information is not available in the provided context." 
    CONTEXT: {context}
    CHAT HISTORY: {chat_history}
    QUESTION: {question}
    ANSWER:
    </s>[INST]
    """
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

# Initialize the Together API
try:
    llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.5,
        max_tokens=1024,
        together_api_key=TOGETHER_AI_API,
    )
except Exception as e:
    logger.error(f"Error initializing Together API: {e}")
    raise RuntimeError("Together API could not be initialized. Check your API key and network connection.")

# Initialize conversational retrieval chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
)

# Initialize FastAPI app
app = FastAPI()


# Define request and response models
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Hello, World!"}


# Chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Pass the user question
        result = qa.invoke(input=request.question)
        answer = result.get("answer", "The chatbot could not generate a response.")
        return ChatResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error during chat invocation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Start Uvicorn server if run directly
if __name__ == "__main__":
    ENV = os.getenv("ENV", "prod")
    PORT = int(os.environ.get("PORT", 8000))

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=(ENV == "dev"))

