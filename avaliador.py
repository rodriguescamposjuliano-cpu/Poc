import json
import time
import re
import math
import traceback
import pandas as pd

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_core.output_parsers import StrOutputParser

# Compatibilidade Ragas
try:
    from ragas.metrics import ResponseRelevancy
    RelevancyMetric = ResponseRelevancy
except ImportError:
    from ragas.metrics import AnswerRelevancy
    RelevancyMetric = AnswerRelevancy


class Avaliador:

    FRASE_RECUSA = "Informação não localizada no regimento enviado."

    def __init__(self, ia, prompt_juiz):
        self.ia = ia
        self.prompt_juiz = prompt_juiz

        self.ragas_llm = LangchainLLMWrapper(ia.llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(ia.embeddings)

        self.metric_faithfulness = Faithfulness(llm=self.ragas_llm)
        self.metric_relevancy = RelevancyMetric(
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings
        )

    # =========================================================
    # HELPERS
    # =========================================================

    def _extrair_nota(self, valor):
        try:
            nota = float(valor)
            return 0.0 if math.isnan(nota) else nota
        except:
            return 0.0

    def _extrair_metricas_ragas(self, resultado):
        try:
            df = resultado.to_pandas()

            f = df["faithfulness"][0] if "faithfulness" in df.columns else 0

            if "answer_relevancy" in df.columns:
                r = df["answer_relevancy"][0]
            elif "response_relevancy" in df.columns:
                r = df["response_relevancy"][0]
            else:
                r = 0

            return self._extrair_nota(f), self._extrair_nota(r)

        except Exception:
            return 0.0, 0.0

    def _avaliar_chunks(self, esperados, gerados):
        set_e = set(esperados)
        set_g = set(gerados)

        iguais = "SIM" if set_e == set_g else "NAO"
        contidos = "SIM" if set_e.issubset(set_g) else "NAO"

        return iguais, contidos

    def _avaliar_juiz(self, pergunta, esperada, obtida):
        chain = self.prompt_juiz | self.ia.llm | StrOutputParser()

        try:
            resultado = chain.invoke({
                "pergunta": pergunta,
                "esperada": esperada,
                "obtida": obtida
            }).strip()

            resultado = resultado.replace("```json", "").replace("```", "").strip()

            match = re.search(r'\{.*\}', resultado, re.DOTALL)
            if match:
                resultado = match.group(0)

            dados = json.loads(resultado)

            return float(dados.get("nota", 0)), dados.get("justificativa", "")

        except Exception as e:
            return 0.0, f"Erro no juiz: {str(e)}"

    def _extrair_resposta_principal(self, texto):
        """
        Remove explicações extras e pega só o núcleo da resposta
        """
        partes = texto.split("\n\n")
        return partes[0].strip()

    def _extrair_contexto_real(self, resposta_obj):
        """
        Usa os documentos reais recuperados (melhor para RAGAS)
        """
        try:
            docs = resposta_obj.get("documentos", [])

            if docs:
                return ["\n\n".join([doc.page_content for doc in docs])]

            return ["sem contexto disponível"]

        except:
            return ["sem contexto disponível"]

    # =========================================================
    # CORE
    # =========================================================

    def avaliar_pergunta(self, item):
        pergunta = item.get("pergunta")
        esperada = item.get("resposta_referencia", "")
        chunks_esperados = item.get("chunks_esperados", [])

        arquivo_origem = item.get("arquivo_origem", "")
        tenant = arquivo_origem.split("_")[0] if arquivo_origem else "001"

        resposta_obj = self.ia.responder(pergunta, tenant)

        # 🔥 IMPORTANTE: sua IA precisa retornar dict
        if isinstance(resposta_obj, dict):
            resposta_bruta = resposta_obj.get("resposta", "")
        else:
            resposta_bruta = resposta_obj

        # --- Parsing resposta ---
        if "📄 Trecho do regimento:" in resposta_bruta:
            partes = resposta_bruta.split("📄 Trecho do regimento:")
            resposta = partes[0].strip()
            trecho = partes[1].split("📎 Fontes:")[0].strip()
        else:
            resposta = resposta_bruta.strip()
            trecho = ""

        # 🔥 Usa resposta limpa (sem ruído)
        resposta_principal = self._extrair_resposta_principal(resposta)

        # --- Extrai chunks ---
        raw_chunks = re.findall(r'doc_\d+#(?:chunk_)?\d+', resposta_bruta)
        chunks = list(set([
            f"{c.split('#')[0]}#chunk_{c.split('#')[1].replace('chunk_', '')}"
            for c in raw_chunks
        ]))

        # 🔥 CONTEXTO REAL (ESSENCIAL)
        contexto = self._extrair_contexto_real(resposta_obj)

        # --- Caso recusa ---
        if resposta == self.FRASE_RECUSA and esperada == self.FRASE_RECUSA:
            f_score, r_score = 1.0, 1.0
            nota_ragas = 10.0
            nota_juiz, justificativa = 10.0, "Recusa correta"
        else:
            dataset = Dataset.from_dict({
                "question": [pergunta],
                "answer": [resposta_principal],
                "contexts": [contexto],
                "ground_truth": [esperada.strip()]
            })

            resultado_ragas = evaluate(
                dataset,
                metrics=[self.metric_faithfulness, self.metric_relevancy],
                show_progress=False
            )

            f_score, r_score = self._extrair_metricas_ragas(resultado_ragas)

            # 🔥 Peso ajustado (relevância > fidelidade)
            nota_ragas = ((f_score * 0.4) + (r_score * 0.6)) * 10

            nota_juiz, justificativa = self._avaliar_juiz(
                pergunta, esperada, resposta
            )

            # 🔥 CALIBRAÇÃO INTELIGENTE
            if nota_juiz >= 9:
                nota_ragas = max(nota_ragas, 8)
            elif nota_juiz >= 7:
                nota_ragas = max(nota_ragas, 6)

        iguais, contidos = self._avaliar_chunks(chunks_esperados, chunks)

        # 🔥 SCORE FINAL HÍBRIDO
        score_final = (nota_ragas * 0.4) + (nota_juiz * 0.6)

        return {
            "Pergunta": pergunta,
            "Esperada": esperada,
            "Obtida": resposta,
            "Chunks Esperados": ", ".join(chunks_esperados) if chunks_esperados else "",
            "Chunks Obtidos": ", ".join(chunks) if chunks else "",
            "Chunks Iguais": iguais,
            "Chunks Contidos": contidos,
            "Fidelidade": f_score,
            "Relevancia": r_score,
            "Nota Ragas": nota_ragas,
            "Nota Juiz": nota_juiz,
            "Score Final": score_final,
            "Justificativa": justificativa
        }

    # =========================================================
    # SINGLE
    # =========================================================

    def rodar_unico(self, caminho_json):
        with open(caminho_json, "r", encoding="utf-8") as f:
            item = json.load(f)

        try:
            if isinstance(item, list):
                item = item[0]

            resultado = self.avaliar_pergunta(item)
            return resultado

        except Exception as e:
            print(f"Erro: {e}")
            traceback.print_exc()
            return None