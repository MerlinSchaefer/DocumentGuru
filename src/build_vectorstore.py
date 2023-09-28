
import box
import yaml
import logging
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings

# load config
with open('./config.yaml', 'r', encoding='utf8') as file:
    config = box.Box(yaml.safe_load(file))


#TODO create wrapper class or function to handle all types of documents
def build_vectorstore(config: Box) -> None:
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

        logger.info("VectorStore created and saved successfully.")

    except Exception as e:
        logger.error(f"An error occurred while building VectorStore: {str(e)}")

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # Set the logging level
    build_vectorstore(config)
