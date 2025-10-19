markdown# 🦜 Sistema RAG (Retrieval Augmented Generation) Local com Llama 3

Este projeto implementa um Sistema de Geração Aumentada por Recuperação (RAG) para consulta a documentos, combinando a arquitetura robusta do LangChain com modelos de código aberto. O sistema utiliza a interface Gradio para permitir a interação com o usuário.

**Foco do Projeto:** Garantir a máxima **portabilidade** (funciona bem em CPU) e **velocidade de resposta** (minimizando o tempo de inferência do LLM).

## 🚀 Tecnologias Principais

* **LLM (Modelo de Linguagem):** [**Llama 3**](https://ollama.com/library/llama3) (Rodando localmente via **Ollama**)
* **Framework:** **LangChain** (para orquestração RAG)
* **Embeddings:** **`all-MiniLM-L6-v2`** (via HuggingFace) – Modelo ultrarrápido, otimizado para execução em CPU
* **Banco de Dados Vetorial:** **Chroma DB** (armazenamento local na pasta `db`)
* **Interface:** **Gradio**

## ⚡ Otimizações para Velocidade (CPU-Only)

Para reduzir a latência de aproximadamente 20 segundos em ambientes com restrição de hardware, foram implementadas as seguintes otimizações no código:

1. **Modelo LLM Leve:** Uso do Llama 3 (versão base ou quantizada) via Ollama
2. **Modelo de Embeddings Rápido:** Uso do `all-MiniLM-L6-v2`
3. **Contexto Reduzido:** A busca de documentos (`similarity_search`) foi limitada a **`k=2`**, enviando menos texto para o LLM processar e acelerando o tempo de geração da resposta

## ⚙️ Pré-requisitos

Para executar o sistema, você precisa ter:

1. **Python 3.9+** instalado
2. O software **[Ollama](https://ollama.com/)** instalado e o serviço **ativo** (rodando em segundo plano)

## 💡 Como Executar o Projeto

### 1. Configurar o Ambiente Python
```bash
# Crie o ambiente virtual
python -m venv .venv

# Ative o ambiente virtual
# No Windows:
.venv\Scripts\activate
# No macOS/Linux:
source .venv/bin/activate

# Instale as dependências listadas no requirements.txt
pip install -r requirements.txt
```

### 2. Baixar o Modelo LLM (Llama 3)

O modelo deve ser instalado no Ollama antes de rodar o código Python:
```bash
ollama pull llama3
```

### 3. Executar o Script Principal

Certifique-se de que o Ollama está ativo e que o banco de dados vetorial (`db/`) já foi criado em uma etapa anterior (por exemplo, usando um script `criar_db.py` separado).

Com o ambiente virtual ativado, execute o script:
```bash
python rag_app.py
```

> **Nota:** Ajuste `rag_app.py` para o nome real do seu arquivo.

A interface Gradio será aberta automaticamente no seu navegador, permitindo que você interaja com o sistema RAG.

## 📁 Estrutura do Projeto
```
.
├── db/                  # Banco de dados vetorial Chroma
├── rag_app.py          # Script principal da aplicação
├── criar_db.py         # Script para criar o banco de dados (opcional)
├── requirements.txt    # Dependências do projeto
└── README.md          # Este arquivo
```

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## 📄 Licença

Este projeto está sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.