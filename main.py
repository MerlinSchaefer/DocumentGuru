import box
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.retrieval_engine import setup_dbqa

if __name__ == "__main__":

    # load config
    with open('./config.yaml', 'r', encoding='utf8') as file:
        config = box.Box(yaml.safe_load(file))

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    start = timeit.default_timer() # Start timer

    # Setup QA object
    dbqa = setup_dbqa()

    # Parse input from argparse into QA object
    response = dbqa({'query': args.input})
    end = timeit.default_timer() # End timer

    # Print document QA response
    print(f'\nAnswer: {response["result"]}')
    print('='*50) # Formatting separator

    # Process source documents for better display
    source_docs = response['source_documents']
    for i, doc in enumerate(source_docs):
        print(f'\nSource Document {i+1}\n')
        print(f'Source Text: {doc.page_content}')
        print(f'Document Name: {doc.metadata["source"]}')
        print(f'Page Number: {doc.metadata["page"]}\n')
        print('='* 50) # Formatting separator
        
    # Display time taken for CPU inference
    print(f"Time to retrieve response: {end - start}")