import os
import sys
import warnings
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")


from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

sys.path.append("src/")

from helper import template
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
        if os.path.exists(self.CONFIG["path"]["PDFs_PATH"]):
            self.loader = DirectoryLoader(
                path=self.CONFIG["path"]["PDFs_PATH"],
                glob="**/*.pdf",
                use_multithreading=True,
                loader_cls=PyPDFLoader,
            )

            self.documents = self.loader.load()

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.CONFIG["token"]["chunk_size"],
                chunk_overlap=self.CONFIG["token"]["chunk_overlap"],
            )

            self.documents = self.text_splitter.split_documents(
                documents=self.documents
            )

            if self.CONFIG["path"]["PROCESSED_PATH"]:
                dump(
                    value=self.documents,
                    filename=os.path.join(
                        self.CONFIG["path"]["PROCESSED_PATH"], "documents.pkl"
                    ),
                )

                print(
                    "Documents file is store in the folder:",
                    self.CONFIG["path"]["PROCESSED_PATH"],
                )

            else:
                raise Exception("The processed path is not defined".capitalize())

        else:
            raise Exception(
                "PDFs path cannot be found to extract the dataset".capitalize()
            )

    def persist_to_database(self):
        if os.path.exists(self.CONFIG["path"]["PROCESSED_PATH"]):

            self.documents = load(
                filename=os.path.join(
                    self.CONFIG["path"]["PROCESSED_PATH"], "documents.pkl"
                ),
            )

            if not os.path.exists(self.CONFIG["path"]["DATABASE_PATH"]):
                os.makedirs(self.CONFIG["path"]["DATABASE_PATH"], exist_ok=True)

            self.persist_directory = self.CONFIG["path"]["DATABASE_PATH"]

            try:
                self.vectordb = Chroma.from_documents(
                    documents=self.documents,
                    embedding=OpenAIEmbeddings(),
                    persist_directory=self.persist_directory,
                )

                print(
                    "Database created and all the documents are stored in the database".title()
                )

            except FileNotFoundError as fnf_error:
                print(f"File not found error: {fnf_error}".capitalize())

            except ValueError as val_error:
                print(f"Value error: {val_error}".capitalize())

            except Exception as e:
                print(f"An unexpected error occurred: {e}".capitalize())

        else:
            raise Exception("The processed path is not defined".capitalize())

    def access_to_db(self):
        self.database = Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory=self.CONFIG["path"]["DATABASE_PATH"],
        )

        return self.database

    def chatReceipe(self):
        self.database = self.access_to_db()

        self.retriever = self.database.as_retriever(
            search_kwargs={"k": self.CONFIG["retriever"]["k"]}
        )

        self.prompt = PromptTemplate(
            input_variables=["context", "question", "history"], template=template
        )
        self.memory = ConversationBufferMemory(
            input_key="question", memory_key="history"
        )

        self.chain = RetrievalQA.from_chain_type(
            llm=OpenAI(
                temperature=1.0,
                model_name="gpt-3.5-turbo",
                openai_api_key=self.access_api_key(),
            ),
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt, "memory": self.memory},
        )

        while True:
            inputs = input("Query: ")
            result = self.chain(inputs=inputs)["result"]

            print("Answer:", result)


if __name__ == "__main__":
    receipe = ReceipeGenerator()
    # receipe.extract_dataset()
    # receipe.persist_to_database()
    # receipe.access_to_db()
    receipe.chatReceipe()
