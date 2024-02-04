from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List
import logging 

def loadPdfPagesFromPath(filePath:str)-> List[Document]:
    """Loads a pdf document for indexing given the path 
    Parameters
    ----------
    filePath : str
        The path of the pdf on the device

    Returns
    -------
        List[Documents]
            Chunks of documents
    """
    logging.info(f"Loading file {filePath}")
    return PyPDFLoader(filePath).load_and_split()


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

def saveEmbeddings(chunks: List[Document],embeddingFunction: SentenceTransformerEmbeddings,embeddingPath:str):
    """"Saves the embeddings locally
    Parameters
    ----------
    chunks : List[Document]
        Chunks of Documents, pages from the PDF and their metadata
    embeddingFunction: SentenceTransformerEmbeddings
        Function used to create embeddings
    embeddingPath: str
        Where to save the embedding db
    """
    logging.info(f"Saving embedding to {embeddingPath}")
    Chroma.from_documents(chunks, embeddingFunction, persist_directory =embeddingPath)

def runIndexing(pdfTokensSource:str, openSourceEmbedding:str, embeddingSavingPath:str):
    """Starts the indexing step given the knowledge source and the embedding algorith name and the path to save the embeddings
    Parameters
    ----------
    pdfTokensSource: path to the pdf source #TODO: allow a list of path in the future
    openSourceEmbedding: str
        The name of the transformer on hugginface
    embeddingSavingPath: str
        Where to save the embedding db
    """
    logging.info(f"Running indexing for {pdfTokensSource} with {openSourceEmbedding} to be saved in {embeddingSavingPath}")
    documentChunks = loadPdfPagesFromPath(pdfTokensSource)
    embeddingFunction = getEmbeddingFunction(openSourceEmbedding)
    saveEmbeddings(chunks=documentChunks, embeddingFunction=embeddingFunction, embeddingPath=embeddingSavingPath)