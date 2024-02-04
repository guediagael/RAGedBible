from langchain_community.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents.base import Document

from typing import List, Dict
import logging

vectorStores : Dict[str,Chroma] = dict()

def getEmbeddingFunction(embeddingName:str):
    """Returns an embedding function given an open source transformer name
    Parameters
    ----------
    embeddingName : str
        The name of the transformer on hugginface

    Returns
    -------
        SentenceTransformerEmbeddings
            The embedding function
    """
    logging.info(f"Loading embedding: {embeddingName}")
    return SentenceTransformerEmbeddings(model_name=embeddingName)


def loadVectorStore(storePath:str, embeddingFunction:SentenceTransformerEmbeddings
)->Chroma:
    logging.info(f"Loading vector store {storePath}")
    global vectorStores
    vectorStore =  Chroma(persist_directory = storePath, embedding_function = embeddingFunction)
    vectorStores[storePath] = vectorStore

def initVectorStore(storePath:str, embeddingFunction:str):
    logging.info(f"Initializing vector embradding {embeddingFunction}")
    embedding = getEmbeddingFunction(embeddingFunction)
    return loadVectorStore(storePath=storePath, embeddingFunction= embedding)


def retrieveDocuments(query:str, storePath:str, embeddingName: str)-> List[Document]:
    logging.info(f"Querying {query}")
    global vectorStores
    if vectorStores.get(storePath) is None:
        initVectorStore(storePath=storePath, embeddingFunction=embeddingName)
    return vectorStores[storePath].similarity_search(query)