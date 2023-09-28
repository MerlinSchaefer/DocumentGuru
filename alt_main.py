import logging
import timeit
import argparse
import yaml
import os
from box import Box
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredMarkdownLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from typing import List
from src.prompts import base_qa_template


class VectorStoreBuilder:
    def __init__(self, config: Box, document_type: str):
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
            texts: Texts = text_splitter.split_documents(documents)

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
    def __init__(self, config_path: str, input_text: str):
        self.config_path = config_path
        self.input_text = input_text
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def load_config(self) -> Box:
        """
        Load configuration parameters from a YAML file.

        Returns:
            Box: A Box object containing the configuration parameters.
        """
        with open(self.config_path, 'r', encoding='utf8') as file:
            config = Box(yaml.safe_load(file))
        return config

    def build_vectorstore(self, config: Box) -> None:
        """
        Build a VectorStore from PDF documents.

        Args:
            config (Box): Configuration parameters including DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, and DB_FAISS_PATH.

        Returns:
            None
        """

        try:
            # Load the files from the data path
            loader = DirectoryLoader(
                path=config.DATA_PATH,
                glob="*.pdf",
                loader_cls=PyPDFLoader
            )

            documents = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            texts: Texts = text_splitter.split_documents(documents)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'}
            )

            vectorstore = FAISS.from_documents(texts, embeddings)
            vectorstore.save_local(config.DB_FAISS_PATH)

            self.logger.info("VectorStore created and saved successfully.")

        except Exception as e:
            self.logger.error(f"An error occurred while building VectorStore: {str(e)}")

    def setup_dbqa(self, config: Box) -> RetrievalQA:
        """
        Setup and configure a RetrievalQA object.

        Args:
            config (Box): Configuration parameters including MODEL_BIN_PATH, MODEL_TYPE, MAX_NEW_TOKENS, TEMPERATURE, and more.

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

    def create_llm(self, config: Box):
        """
        Create a Language Model.

        Args:
            config (Box): Configuration parameters including MODEL_BIN_PATH, MODEL_TYPE, MAX_NEW_TOKENS, TEMPERATURE, and more.

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
        config = self.load_config()
        dbqa = self.setup_dbqa(config)

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
            print(f'Page Number: {doc.metadata["page"]}\n')
            print('=' * 50)  # Formatting separator

        # Display time taken for CPU inference
        end = timeit.default_timer()  # End timer
        print(f"Time to retrieve response: {end - start}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--build-vectorstore', action='store_true',
                        help="Specify to build VectorStore. If not specified, it won't build VectorStore.")
    args = parser.parse_args()

    config_path = './config.yaml'
    input_text = args.input

    # Initialize the DocumentProcessor
    processor = DocumentProcessor(config_path, input_text)

    if args.build_vectorstore:
        # Check if the vectorstore directory exists
        vectorstore_dir = './vectorstores/'
        if not os.path.exists(vectorstore_dir):
            logging.warning(f"'{vectorstore_dir}' directory not found. Building VectorStore is probably needed. Use '--build-vectorstore'.")

        # Build VectorStore
        processor.build_vectorstore(processor.load_config())

    # Process documents
    processor.process_documents()
