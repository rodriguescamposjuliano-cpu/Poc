import json
import time
import os
import re
import pandas as pd
from google import genai
from dotenv import load_dotenv

# =====================================================================
# 1. SETUP E CARREGAMENTO
# =====================================================================
try:
    from main import SeverinoIA
except ImportError:
    print("Aviso: Módulo 'main.SeverinoIA' não encontrado. Verifique se o arquivo main.py está na pasta.")

load_dotenv()
CHAVE_API = os.environ.get("GEMINI_API_KEY")

if not CHAVE_API:
    print("ERRO CRÍTICO: Chave GEMINI_API_KEY não encontrada no .env!")
    exit(1)

# Inicializa o cliente Gemini
client = genai.Client(api_key=CHAVE_API)
MODELO_JUIZ = 'gemini-2.5-flash'

def avaliar_conteudo_com_llm(pergunta, resposta_esperada, resposta_gerada):
    """Usa o Gemini para atribuir uma nota de 0 a 10 à resposta gerada."""
    prompt = f"""
    Você é um avaliador especialista de sistemas RAG.
    Sua tarefa é comparar a RESPOSTA DO SISTEMA com o GABARITO e dar uma nota de 0 a 10.

    Pergunta: {pergunta}
    Gabarito Esperado: {resposta_esperada}
    Resposta do Sistema: {resposta_gerada}

    Responda EXATAMENTE neste formato:
    [NOTA] número
    [JUSTIFICATIVA] texto (máximo 2 linhas)
    """
    try:
        response = client.models.generate_content(model=MODELO_JUIZ, contents=prompt)
        texto = response.text
        
        # Regex flexível para capturar a nota (mesmo com Markdown ou colchetes)
        match_nota = re.search(r"NOTA\]?[:\s*]*(\d+)", texto.upper())
        nota = int(match_nota.group(1)) if match_nota else 0
        
        # Extração da justificativa
        justificativa = texto.split("[JUSTIFICATIVA]")[-1].strip() if "[JUSTIFICATIVA]" in texto else texto
        return nota, justificativa
    except Exception as e:
        print(f"Erro no LLM Avaliador: {e}")
        return 0, str(e)

def avaliar_chunks(chunks_esperados, chunks_gerados):
    """Compara se os chunks recuperados são iguais aos esperados."""
    set_esperados = set(chunks_esperados)
    set_gerados = set(chunks_gerados)
    sao_iguais = "SIM" if set_esperados == set_gerados else "NAO"
    return sao_iguais, list(set_gerados)

def main():
    print("="*50)
    print("INICIANDO MOTOR DE AVALIAÇÃO SEVERINO IA")
    print("="*50)
    
    try:
        ia = SeverinoIA()
    except Exception as e:
        print(f"Erro ao inicializar SeverinoIA: {e}")
        return

    try:
        with open('golden_set.json', 'r', encoding='utf-8') as f:
            golden_set = json.load(f)
    except FileNotFoundError:
        print("Erro: Arquivo 'golden_set.json' não encontrado.")
        return

    resultados_finais = []
    arquivo_saida = 'resultado_avaliacao.xlsx'
    
    for item in golden_set:
        id_pergunta = item.get('id')
        pergunta = item.get('pergunta')
        resposta_esperada = item.get('resposta_referencia', '')
        chunks_esperados = item.get('chunks_esperados', [])
        
        print(f"\n>>> [ID {id_pergunta}] PROCESSANDO PERGUNTA...")
        
        try:
            # Identifica o tenant pelo nome do arquivo no golden_set
            arquivo_origem = item.get('arquivo_origem', '')
            tenant_id = arquivo_origem.split('_')[0] if arquivo_origem else "001"
            
            # 1. Executa o seu sistema RAG
            resposta_bruta = ia.responder(pergunta, tenant_id)
            
            # 2. Extrai a resposta limpa (removendo os metadados de chunks se houver)
            if "📄 Trecho do regimento:" in resposta_bruta:
                resposta_gerada = resposta_bruta.split("📄 Trecho do regimento:")[0].strip()
            else:
                resposta_gerada = resposta_bruta.strip()

            # 3. Extrai e normaliza os chunks informados pelo sistema
            raw_chunks = re.findall(r'doc_\d+#(?:chunk_)?\d+', resposta_bruta)
            chunks_informados = list(set([f"{c.split('#')[0]}#chunk_{c.split('#')[1].replace('chunk_', '')}" for c in raw_chunks]))
            
            # 4. Avaliação do Conteúdo (Nota 0-10)
            nota, justificativa = avaliar_conteudo_com_llm(pergunta, resposta_esperada, resposta_gerada)
            
            # 5. Avaliação dos Chunks
            correspondencia, lista_chunks_inf = avaliar_chunks(chunks_esperados, chunks_informados)
            
            # --- EXIBIÇÃO NO CONSOLE (DETALHADA) ---
            print(f"❓ Pergunta: {pergunta}")
            print(f"🎯 Resposta Esperada: {resposta_esperada}")
            print(f"🤖 Resposta Obtida: {resposta_gerada}")
            print(f"📂 Chunks Esperados: {chunks_esperados}")
            print(f"📂 Chunks Informados: {chunks_informados}")
            print(f"⚖️  Nota: [{nota}/10]")
            print(f"💬 Justificativa: {justificativa}")
            print(f"🔗 Correspondência Chunks: {correspondencia}")
            print("-" * 60)

            # --- ARMAZENAMENTO PARA O EXCEL ---
            resultados_finais.append({
                'ID': id_pergunta,
                'Pergunta': pergunta,
                'Categoria': item.get('categoria', 'N/A'),
                'Resposta Esperada': resposta_esperada,
                'Resposta Obtida': resposta_gerada,
                'Nota Conteúdo': nota,
                'Justificativa': justificativa,
                'Chunks Esperados': ", ".join(chunks_esperados),
                'Chunks Informados': ", ".join(chunks_informados),
                'Chunks Iguais': correspondencia
            })
            
            # Respeita o Rate Limit da API
            time.sleep(3)

        except Exception as e:
            print(f"[!] Erro Crítico no ID {id_pergunta}: {e}")

    # Salva o arquivo final em Excel
    if resultados_finais:
        df = pd.DataFrame(resultados_finais)
        df.to_excel(arquivo_saida, index=False)
        print(f"\n✅ AVALIAÇÃO FINALIZADA COM SUCESSO!")
        print(f"📊 Planilha gerada: {arquivo_saida}")
    else:
        print("\n⚠️ Nenhum dado foi processado para salvar.")

if __name__ == "__main__":
    main()