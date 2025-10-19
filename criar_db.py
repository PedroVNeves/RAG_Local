from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#Modelo escolhido para o embedding
model_name = "all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    # O dispositivo 'cpu' garante que funcione em qualquer lugar
    model_kwargs={'device': 'cpu'}
)


#Diretório da pasta com os arquivos que servirão de base
PASTA_BASE = "base"

#função para criar o banco de dados separado em chunks vetorizados
def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)
    
#função para carregars os documentos da pasta base
def carregar_documentos():
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
    documentos = carregador.load()
    return documentos

#função para dividir os documentos em chunks
def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size =1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True
    )
    chunk = separador_documentos.split_documents(documentos)
    return chunk

#função para vetorizar os chunks e criar o db
def vetorizar_chunks(chunks):
    db = Chroma.from_documents(chunks, embedding_model, persist_directory= "db")
criar_db()