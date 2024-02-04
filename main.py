from index import runIndexing
from context import retrieveDocuments

import os
import logging

KNOWLEDGE_SOURCE_ROOT_PATH = 'knowledge_base'
EMBEDDINGS_ROOT_PATH = 'embeddings'
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    knowledgeSource = [el for el in os.listdir('knowledge_base') if el.endswith('.pdf')]
    logging.info(f"Found {len(knowledgeSource)} files in knowledge source")
    if len(knowledgeSource)>0:
        for source in knowledgeSource:
            embeddingsPath = f'{EMBEDDINGS_ROOT_PATH}/{source.split(".")[0]}'
            embeddingsAlgorithm = 'all-MiniLM-L6-v2'
            # runIndexing(pdfTokensSource=f'{KNOWLEDGE_SOURCE_ROOT_PATH}/{source}', openSourceEmbedding=embeddingsAlgorithm,embeddingSavingPath=embeddingsPath)
            print(f"Retrieving from {source}")
            results = retrieveDocuments(query="Should I hate someone who killed my parents?",storePath = embeddingsPath, embeddingName=embeddingsAlgorithm)
            for i in range(len(results)):
                print(f"Result {i+1} In page {results[i].metadata['page']}, it says {results[i].page_content}")