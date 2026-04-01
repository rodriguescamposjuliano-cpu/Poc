import json
import csv
import time
import os
import re
from google import genai
from dotenv import load_dotenv

# =====================================================================
# 1. IMPORTE AQUI OS MÓDULOS DO SEU SISTEMA
# =====================================================================
# Importa a classe principal do seu RAG
from main import SeverinoIA

# Carrega as variáveis do arquivo .env para o sistema
load_dotenv()

# =====================================================================
# 2. CONFIGURAÇÃO DO LLM AVALIADOR (NOVO SDK)
# =====================================================================
CHAVE_API = os.environ.get("GEMINI_API_KEY")

if not CHAVE_API:
    print("ERRO CRÍTICO: Chave GEMINI_API_KEY não encontrada no .env!")
    exit(1)

# Inicializa o cliente da nova biblioteca
client = genai.Client(api_key=CHAVE_API)

# Define o modelo avaliador (2.5-flash é rápido e barato para avaliações)
MODELO_JUIZ = 'gemini-2.5-flash'

def avaliar_conteudo_com_llm(pergunta, resposta_esperada, resposta_gerada):
    """Usa o Gemini para verificar se a resposta gerada bate semanticamente com o gabarito."""
    prompt = f"""
    Você é um avaliador de um sistema de Inteligência Artificial (RAG).
    Sua tarefa é comparar a resposta gerada pelo sistema com o gabarito de referência. Caso a resposta gerada aborde de forma ampla o conteúdo do gabarito, mesmo que não seja idêntica, considere como correta. Respostas que não abordem os pontos principais do gabarito ou que sejam vazias devem ser consideradas incorretas.

    Pergunta: {pergunta}
    Gabarito Esperado: {resposta_esperada}
    Resposta do Sistema: {resposta_gerada}

    A resposta do sistema aborda corretamente os pontos do gabarito esperado? Considere que respostas vazias ou "não encontrei a informação" quando havia gabarito devem ser reprovadas.
    
    Responda ESTRITAMENTE no formato abaixo:
    [AVALIACAO] (SIM ou NAO)
    [JUSTIFICATIVA] (Sua explicação breve de no máximo 2 linhas)
    """
    try:
        response = client.models.generate_content(
            model=MODELO_JUIZ,
            contents=prompt
        )
        texto = response.text
        
        # Extração simples das tags
        avaliacao = "SIM" if "[AVALIACAO] SIM" in texto.upper() else "NAO"
        justificativa = texto.split("[JUSTIFICATIVA]")[-1].strip() if "[JUSTIFICATIVA]" in texto else texto
        
        return avaliacao, justificativa
    except Exception as e:
        print(f"Erro ao chamar LLM Avaliador: {e}")
        return "ERRO", str(e)

def avaliar_chunks(chunks_esperados, chunks_gerados):
    """Compara se os chunks recuperados são iguais aos esperados."""
    set_esperados = set(chunks_esperados)
    set_gerados = set(chunks_gerados)
    
    sao_iguais = (set_esperados == set_gerados)
    return "SIM" if sao_iguais else "NAO", list(set_gerados)

def main():
    # =================================================================
    # 3. INICIALIZAÇÃO DO SEU SISTEMA
    # =================================================================
    print("Inicializando o motor do Severino IA para avaliação...")
    ia = SeverinoIA()
    
    # Carrega o Golden Set
    try:
        with open('golden_set.json', 'r', encoding='utf-8') as f:
            golden_set = json.load(f)
    except FileNotFoundError:
        print("ERRO: Arquivo 'golden_set.json' não encontrado na pasta atual.")
        return
        
    print(f"\nIniciando avaliação de {len(golden_set)} perguntas...")
    
    # Prepara o arquivo CSV
    arquivo_csv = 'resultado_avaliacao.csv'
    colunas = ['id', 'pergunta', 'categoria', 'resposta_esperada', 'resposta_gerada', 
               'conteudo_aprovado', 'justificativa_conteudo', 
               'chunks_esperados', 'chunks_gerados', 'chunks_iguais']
    
    with open(arquivo_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=colunas)
        writer.writeheader()
        
        for item in golden_set:
            id_pergunta = item.get('id')
            pergunta = item.get('pergunta')
            resposta_esperada = item.get('resposta_referencia', '')
            chunks_esperados = item.get('chunks_esperados', [])
            
            print(f"\n[{id_pergunta}] Processando: {pergunta}")
            
            sucesso = False
            tentativas = 0
            
            while not sucesso and tentativas < 3:
                try:
                    # =========================================================
                    # 4. CHAMADA AO SEU SISTEMA E EXTRAÇÃO DE DADOS
                    # =========================================================
                    arquivo = item.get('arquivo_origem', '')
                    tenant_id = arquivo.split('_')[0] if arquivo else "001"
                    
                    # 1) Chamada oficial ao método
                    resposta_bruta = ia.responder(pergunta, tenant_id)
                    
                    # 2) Extração da Resposta Principal (antes da tag '📄 Trecho do regimento:')
                    if "📄 Trecho do regimento:" in resposta_bruta:
                        resposta_gerada = resposta_bruta.split("📄 Trecho do regimento:")[0].strip()
                    else:
                        resposta_gerada = resposta_bruta.strip()

                    # 3) Extração e Normalização dos Chunks
                    raw_chunks = re.findall(r'doc_\d+#(?:chunk_)?\d+', resposta_bruta)
                    
                    chunks_gerados = []
                    for c in raw_chunks:
                        partes = c.split('#')
                        doc_part = partes[0]
                        chunk_part = partes[1].replace('chunk_', '')
                        chunks_gerados.append(f"{doc_part}#chunk_{chunk_part}")
                        
                    chunks_gerados = list(set(chunks_gerados))
                    # =========================================================
                    
                    # Realiza Avaliações
                    conteudo_aprovado, justificativa = avaliar_conteudo_com_llm(pergunta, resposta_esperada, resposta_gerada)
                    chunks_iguais, lista_chunks_gerados = avaliar_chunks(chunks_esperados, chunks_gerados)
                    
                    # Monta e grava a linha no CSV
                    linha = {
                        'id': id_pergunta,
                        'pergunta': pergunta,
                        'categoria': item.get('categoria'),
                        'resposta_esperada': resposta_esperada,
                        'resposta_gerada': resposta_gerada,
                        'conteudo_aprovado': conteudo_aprovado,
                        'justificativa_conteudo': justificativa,
                        'chunks_esperados': str(chunks_esperados),
                        'chunks_gerados': str(lista_chunks_gerados),
                        'chunks_iguais': chunks_iguais
                    }
                    writer.writerow(linha)
                    csvfile.flush() # Salva no disco imediatamente
                    
                    # --- EXIBIÇÃO NO CONSOLE ---
                    print(f" 🎯 Esperado: {resposta_esperada}")
                    print(f" 🤖 Gerado:   {resposta_gerada}")
                    print(f" -> Avaliação Conteúdo: [{conteudo_aprovado}] - {justificativa}")
                    print(f" -> Avaliação Chunks:   [{chunks_iguais}] (Esperava: {chunks_esperados} | Trouxe: {lista_chunks_gerados})")
                    print("-" * 80)
                    
                    sucesso = True
                    
                    # Pausa de 10 segundos para não estourar limite da API (Rate Limit)
                    time.sleep(10)
                    
                except Exception as e:
                    tentativas += 1
                    print(f"\n[!] Falha na execução da pergunta {id_pergunta} (Tentativa {tentativas}/3): {e}")
                    if tentativas < 3:
                        print("Aguardando 5 segundos para tentar novamente...")
                        time.sleep(5)
                    else:
                        print("Muitas falhas. Pulando esta pergunta.")

    print(f"\n✅ Avaliação finalizada com sucesso! Resultados salvos em '{arquivo_csv}'.")

if __name__ == "__main__":
    main()