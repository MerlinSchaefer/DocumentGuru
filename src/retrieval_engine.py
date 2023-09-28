from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.prompts import base_qa_template
from src.llm import create_llm

# Wrap prompt template in a PromptTemplate object
def setup_qa_prompt(qa_template):
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt

# Build RetrievalQA object
def create_retrieval_qa(llm, prompt, vectordb, return_source_docs = True):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k':2}),
                                       return_source_documents=return_source_docs,
                                       chain_type_kwargs={'prompt': prompt})
    return dbqa

# Instantiate QA object
def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    qa_prompt = setup_qa_prompt(qa_template=base_qa_template)
    llm = create_llm()
    dbqa = create_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa