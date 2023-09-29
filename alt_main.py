import logging
import timeit
import argparse
import yaml
import os
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredMarkdownLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.schema import Document
from typing import List
from src.prompts import base_qa_template

class Config(BaseModel):
    RETURN_SOURCE_DOCUMENTS: bool
    VECTOR_COUNT: int
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    DATA_PATH: str
    DB_FAISS_PATH: str
    MODEL_TYPE: str
    MODEL_BIN_PATH: str
    MAX_NEW_TOKENS: int
    TEMPERATURE: float
    # Add other configuration parameters as needed

    @classmethod
    def load_from_file(cls, config_path: str) -> 'Config':
        """
        Load configuration parameters from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            Config: An instance of the Config class with loaded configuration parameters.
        """
        with open(config_path, 'r', encoding='utf8') as file:
            config_data = yaml.safe_load(file)
            return cls(**config_data)


class VectorStoreBuilder:
    def __init__(self, config: Config, document_type: str):
        self.config = config
        self.document_type = document_type
        self.logger = logging.getLogger(__name__)


    
    def build_vectorstore(self):
        try:
            # Set loader_cls and glob based on document_type
            if self.document_type == "pdf":
                loader_cls = PyPDFLoader
                glob = "*.pdf"
            elif self.document_type == "md":
                loader_cls = UnstructuredMarkdownLoader
                glob = "*.md"
            else:
                raise ValueError("Unsupported document_type. Use 'pdf' or 'md'.")

            # Load the files from the data path with the specified loader
            loader = DirectoryLoader(
                path=self.config.DATA_PATH,
                glob=glob,
                loader_cls=loader_cls
            )

            documents = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            texts: List[Document] = text_splitter.split_documents(documents)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'}
            )

            vectorstore = FAISS.from_documents(texts, embeddings)
            vectorstore.save_local(self.config.DB_FAISS_PATH)

            self.logger.info("VectorStore created and saved successfully.")

        except Exception as e:
            self.logger.error(f"An error occurred while building VectorStore: {str(e)}")


class DocumentProcessor:
    def __init__(self, config: Config, input_text: str):
        self.config= config
        self.input_text = input_text
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)




    def setup_dbqa(self, config: Config) -> RetrievalQA:
        """
        Setup and configure a RetrievalQA object.

        Args:
            config (Config): Configuration parameters including MODEL_BIN_PATH, MODEL_TYPE, MAX_NEW_TOKENS, TEMPERATURE, and more.

        Returns:
            RetrievalQA: A configured RetrievalQA object.
        """
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': 'cpu'})
        vectordb = FAISS.load_local(config.DB_FAISS_PATH, embeddings)
        qa_prompt = self.setup_qa_prompt()
        llm = self.create_llm(config)
        dbqa = self.create_retrieval_qa(llm, qa_prompt, vectordb)

        return dbqa

    def setup_qa_prompt(self) -> PromptTemplate:
        """
        Setup a QA prompt template.

        Returns:
            PromptTemplate: A configured PromptTemplate object.
        """
        prompt = PromptTemplate(template=base_qa_template,
                                input_variables=['context', 'question'])
        return prompt

    def create_llm(self, config: Config):
        """
        Create a Language Model.

        Args:
            config (Config): Configuration parameters including MODEL_BIN_PATH, MODEL_TYPE, MAX_NEW_TOKENS, TEMPERATURE, and more.

        Returns:
            CTransformers: A configured CTransformers Language Model.
        """
        llm = CTransformers(model=config.MODEL_BIN_PATH,
                            model_type=config.MODEL_TYPE,
                            config={'max_new_tokens': config.MAX_NEW_TOKENS,
                                    'temperature': config.TEMPERATURE}
                            )
        return llm

    def create_retrieval_qa(self, llm, prompt, vectordb, return_source_docs=True):
        """
        Create a RetrievalQA object.

        Args:
            llm: Language model.
            prompt: QA prompt template.
            vectordb: Vector database.
            return_source_docs: Whether to return source documents in the response.

        Returns:
            RetrievalQA: A configured RetrievalQA object.
        """
        dbqa = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=return_source_docs,
                                           chain_type_kwargs={'prompt': prompt})
        return dbqa

    def process_documents(self):
        
        dbqa = self.setup_dbqa(self.config)

        start = timeit.default_timer() # Start timer
        # Parse input text into QA object
        response = dbqa({'query': self.input_text})

        # Print document QA response
        print(f'\nAnswer: {response["result"]}')
        print('=' * 50)  # Formatting separator

        # Process source documents for better display
        source_docs = response['source_documents']
        for i, doc in enumerate(source_docs):
            print(f'\nSource Document {i + 1}\n')
            print(f'Source Text: {doc.page_content}')
            print(f'Document Name: {doc.metadata["source"]}')
            print(f'Page Number: {doc.metadata["page"]}\n') # bug if no pages
            print('=' * 50)  # Formatting separator

        # Display time taken for CPU inference
        end = timeit.default_timer()  # End timer
        print(f"Time to retrieve response: {end - start}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--build-vectorstore', action='store_true',
                        help="Specify to build VectorStore. If not specified, it won't build VectorStore.")
    parser.add_argument('--document-type', type=str, choices=["pdf", "md"], default="pdf",
                        help="Specify the type of documents to process ('pdf' or 'md'). Default is 'pdf'.")
    args = parser.parse_args()

    config_path = "./config.yaml"
    config = Config.load_from_file(config_path)
    input_text = args.input

    # Initialize the DocumentProcessor
    processor = DocumentProcessor(config, input_text)



    if args.build_vectorstore:
        # Initialize the VectorStoreBuilder
        vectorstore_builder = VectorStoreBuilder(config, args.document_type)

        # Build VectorStore
        vectorstore_builder.build_vectorstore()
    else:
        # Check if the vectorstore directory exists
        vectorstore_dir = './vectorstores/'
        if not os.path.exists(vectorstore_dir):
            logging.warning(f"'{vectorstore_dir}' directory not found. Building VectorStore is probably needed. Use '--build-vectorstore'.")


    # Process documents
    processor.process_documents()
