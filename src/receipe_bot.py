import os
import warnings
from dotenv import load_dotenv

load_dotenv()


from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ReceipeGenerator:
    def __init__(self, name="Receipe Generator"):
        self.name = name.capitalize()

    def access_api_key(self):
        try:
            return os.getenv("OPENAI_API_KEY")

        except Exception as e:
            print("An error occurred while loading the API key:", e)
            raise


if __name__ == "__main__":
    receipe = ReceipeGenerator()
    print(receipe.access_api_key())
