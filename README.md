# Severino IA - Assistente de Normas Condominiais (RAG)

O **Severino IA** é um ecossistema de Inteligência Artificial baseado em RAG (Geração Aumentada por Recuperação) projetado para responder dúvidas de moradores sobre Regimentos Internos. O sistema utiliza busca híbrida e re-ranqueamento para garantir precisão jurídica e evitar alucinações.


## 🚀 Tecnologias Utilizadas

* **LLM:** Gemini-2.5-Flash.
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
git clone https://github.com/odriguescamposjuliano-cpu/Poc.git
cd Poc
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

## 📂 Estrutura do Projeto
* `indexador.py`: Pipeline de ingestão, OCR, limpeza e criação do banco vetorial.

* `main.py`: Chatbot com motor de busca híbrida e interface CLI.

* `localclassifier.py`: Classificador local para categorização de intenções.

* `treinador.py`: Script para treinar o modelo de classificação local.

* `validador_extracao.py`: Gera relatórios para auditoria dos textos extraídos.

* `regimentos_input/`: Pasta para depositar os PDFs dos condomínios.

## 🏃 Como Executar

1. **Ingestão e Indexação**
Coloque os arquivos PDF na pasta `regimentos_input/` (o nome do arquivo deve começar com o ID do condomínio, ex: `001_Regimento.pdf`) e execute:

```Bash
python indexador.py
```

Este comando criará a pasta `chroma_db/` com os dados processados.


2. **Treino do Classificador Local**

Para que o sistema consiga categorizar as perguntas sem custo de API:

```Bash
python treinador.py
```

3. **Execução do Chatbot**

Inicie a interface de conversação:

```Bash
python main.py
```

4. **Auditoria (Opcional)**

Para conferir se o OCR e a limpeza foram realizados corretamente:

```Bash
python validador_extracao.py
```

## ⚖️ Licença

Este projeto foi desenvolvido como requisito para a disciplina de Processamento de Linguagem Natural da Pós-Graduação em IA do IFG.
