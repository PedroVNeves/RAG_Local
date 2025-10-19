from langchain_chroma.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import gradio as gr

# --- Configurações do RAG ---
#Modelo escolhido para o embedding
model_name = "all-MiniLM-L6-v2"

#configuraçaõ do embedding
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    # O dispositivo 'cpu' garante que funcione em qualquer lugar
    model_kwargs={'device': 'cpu'}
)

#Diretório do DataBase
CAMINHO_DB = "db"

#Modelo do prompt
prompt_template = """
Responda a pergunta do usuário:
{pergunta}

de forma curta e com base nessas informações:

{base_conhecimento}

Se você não encontrar a resposta para a pergunta do usuário nessas informações,
responda não sei te dizer isso
"""

# Inicializa o modelo Ollama e o Banco de Dados Chroma fora da função para eficiência
try:
    # Tenta inicializar o LLM. Garanta que o Ollama esteja rodando e o modelo 'llama3' esteja instalado.
    modelo = OllamaLLM(model="llama3")
except Exception as e:
    print(f"AVISO: Falha ao inicializar Ollama. Verifique se o servidor está rodando e o modelo 'llama3' está instalado. Erro: {e}")
    # Se falhar, defina como None para evitar que a interface trave
    modelo = None

try:
    # Tenta carregar o DB. Garanta que o DB já foi criado com 'criar_db.py'.
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=embedding_model)
except Exception as e:
    print(f"AVISO: Falha ao carregar Chroma DB. Verifique se o diretório '{CAMINHO_DB}' existe. Erro: {e}")
    db = None

# ----------------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL: Recebe a pergunta do Gradio e retorna a resposta
# ----------------------------------------------------------------------------------
def obter_resposta_rag(pergunta):
    if db is None:
        return "Erro: O banco de dados vetorial (Chroma DB) não pôde ser carregado. Verifique os logs."
    if modelo is None:
        return "Erro: O modelo de Linguagem (Ollama LLM) não pôde ser inicializado. Verifique os logs."

    # 1. Busca por similaridade
    # Abaixamos o k para 2 ou 3 para evitar contextos muito grandes
    resultados = db.similarity_search_with_relevance_scores(pergunta, k=3)

    # 2. Verificação de Relevância
    LIMITE_RELEVANCIA = 0.6

    if not resultados or resultados[0][1] > LIMITE_RELEVANCIA:
        return "Não consegui encontrar uma resposta relevante na base de conhecimento. Tente reformular sua pergunta."

    # 3. Formatação da Base de Conhecimento
    textos_resultado = []
    for resultado in resultados:
        # Adicionamos a pontuação para fins de depuração, mas não no prompt final
        # print(f"Resultado encontrado com score: {resultado[1]}")
        textos_resultado.append(resultado[0].page_content)
    
    base_conhecimento = "\n\n----\n\n".join(textos_resultado)
    
    # 4. Construção do Prompt (usando o Template de Chat)
    prompt_final = ChatPromptTemplate.from_template(prompt_template)
    prompt_formatado = prompt_final.invoke({"pergunta": pergunta, "base_conhecimento": base_conhecimento})
    
    # 5. Invocação do Modelo
    # Convertemos o objeto 'prompt_formatado' para uma string para o modelo Ollama
    texto_resposta = modelo.invoke(prompt_formatado.to_string())
    
    return texto_resposta

# ----------------------------------------------------------------------------------
# INTERFACE GRÁFICA COM GRADIO
# ----------------------------------------------------------------------------------

# Criação dos componentes da interface
iface = gr.Interface(
    # A função Python que será chamada
    fn=obter_resposta_rag,
    
    # Componente de entrada (uma caixa de texto para a pergunta)
    inputs=gr.Textbox(
        lines=2, 
        placeholder="Digite sua pergunta aqui...", 
        label="Sua Pergunta"
    ),
    
    # Componente de saída (uma caixa de texto para a resposta da IA)
    outputs=gr.Textbox(
        lines=10, 
        label="Resposta da IA (RAG)"
    ),
    
    # Título e descrição da interface
    title="🤖 Sistema de Perguntas e Respostas (RAG) Local",
    description="Faça uma pergunta e a IA responderá com base no conteúdo da sua base de conhecimento (Arquivos PDF).",
    
    # Tema mais escuro para um visual mais moderno (opcional)
    theme=gr.themes.Soft() 
)

# Inicia a interface. Ela abrirá automaticamente no seu navegador.
iface.launch(share=True)