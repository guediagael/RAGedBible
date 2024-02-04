from index import runIndexing
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
            runIndexing(pdfTokensSource=f'{KNOWLEDGE_SOURCE_ROOT_PATH}/{source}', openSourceEmbedding='all-MiniLM-L6-v2',embeddingSavingPath=embeddingsPath)