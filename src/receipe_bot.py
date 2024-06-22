import os
import sys
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

sys.path.append("src/")

from utils import load, dump, config


class ReceipeGenerator:
    def __init__(self, name="Receipe Generator"):
        self.name = name.capitalize()

        self.CONFIG = config()

    def access_api_key(self):
        try:
            return os.getenv("OPENAI_API_KEY")

        except Exception as e:
            print("An error occurred while loading the API key:", e)
            raise

    def extract_dataset(self):
        if os.path.exists(self.CONFIG["PDF_PATH"]):
            self.loader = DirectoryLoader(
                path=self.CONFIG["PDFs_PATH"], glob="**/*.pdf", use_multithreading=True
            )

            self.documents = self.loader.load()

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CONFIG["token"]["chunk_size"],
                chunk_overlap=self.CONFIG["token"]["chunk_overlap"],
            )

            self.documents = self.text_splitter.split_documents(
                documents=self.documents
            )

        else:
            raise Exception(
                "PDFs path cannot be found to extract the dataset".capitalize()
            )


if __name__ == "__main__":
    receipe = ReceipeGenerator()
    receipe.extract_dataset()
