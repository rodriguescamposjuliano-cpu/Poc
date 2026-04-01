# Severino IA - Assistente de Normas Condominiais (RAG)


O **Severino IA** é um ecossistema de Inteligência Artificial baseado em RAG (Geração Aumentada por Recuperação) projetado para responder dúvidas de moradores sobre Regimentos Internos. O sistema utiliza busca híbrida e re-ranqueamento para garantir precisão jurídica e evitar alucinações.


## 🚀 Tecnologias Utilizadas

* **LLM:** Llama-3.1-8b via Groq.
* **Vector** Database: ChromaDB.
* **Embeddings:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
* **Busca Híbrida:** BM25 (Léxica) + Vetorial (Semântica).
* **Reranker:** FlashRank (ms-marco-MiniLM-L-12-v2).
* **OCR:** PyMuPDF e Tesseract OCR.

## 📋 Pré-requisitos

Antes de iniciar, você precisará ter instalado:

1. **Python 3.10 ou superior**.

2. **Tesseract OCR:**
    * macOS: `brew install tesseract tesseract-lang`
    * Linux: `sudo apt install tesseract-ocr libtesseract-dev`

3. **Poppler (para conversão de PDF):**
    * macOS: `brew install poppler`
    * Linux: `sudo apt install poppler-utils`


## 🛠️ Instalação

1. Clone o repositório:

```Bash
git clone https://github.com/lucasoliveirasouza/poc-severino-ia.git
cd poc-severino-ia
```

2. Crie e ative um ambiente virtual:

```Bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:

```Bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
Crie um arquivo .env na raiz do projeto e adicione sua chave da API Groq:


```Snippet de código
GROQ_API_KEY=sua_chave_aqui
```