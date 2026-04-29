from langchain_community.document_loaders import PyPDFLoader
from app.constant.config import CHUNKING_SIZE, CHUNKING_OVERLAP
from langchain_text_splitters import RecursiveCharacterTextSplitter




def chunk_document(path):
    loader = PyPDFLoader(path)
    return loader.load_and_split()


def chunking_strategy():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNKING_SIZE,
        chunk_overlap=CHUNKING_OVERLAP,
        separators=["\n\n", "\n", ".", " ", "" , "-", "\t"],  # Add more separators for better splitting
        length_function=len,
    )


def chunking_text(text:str):
    return chunking_strategy().split_text(text)
