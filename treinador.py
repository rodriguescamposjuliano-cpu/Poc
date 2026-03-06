import joblib
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. DICIONÁRIOS DE EXPANSÃO (Onde a mágica acontece)
SINONIMOS = {
    "vizinho": ["o morador do lado", "alguém", "o pessoal do 402", "um cara", "o vizinho"],
    "cachorro": ["pet", "animal", "cão", "pitbull", "shitzu", "bicho"],
    "carro": ["veículo", "automóvel", "carro", "caminhonete", "SUV"],
    "barulho": ["som alto", "gritaria", "algazarra", "baderna", "barulheira"],
    "vaga": ["garagem", "estacionamento", "minha vaga", "lugar de parar"]
}

# 2. DATASET BASE (Reduzido para templates)
TEMPLATES = [
    ("{vizinho} está com {barulho}", "Barulho"),
    ("{barulho} vindo do apartamento de cima", "Barulho"),
    ("tem um {cachorro} latindo muito", "Animais"),
    ("sujeira de {cachorro} no elevador", "Animais"),
    ("{vizinho} parou o {carro} na minha {vaga}", "Garagem"),
    ("tem um {carro} estranho na {vaga}", "Garagem"),
    ("lixo jogado no corredor", "Lixo"),
    ("cheiro ruim no duto de lixo", "Lixo"),
    ("quero reservar o salão de festas", "Reservas"),
    ("pode usar a churrasqueira no domingo?", "Reservas"),
    ("transcreva o artigo do regimento", "Transcrição"),
    ("copie o texto da convenção", "Transcrição"),
    ("bom dia severino", "Cordialidade"),
    ("obrigado pela ajuda", "Cordialidade")
]

def gerar_dados_em_massa(n_por_template=100):
    dataset_expandido = []
    for template, categoria in TEMPLATES:
        for _ in range(n_por_template):
            # Substitui os placeholders {chaves} por sinônimos aleatórios
            frase = template
            for chave, opcoes in SINONIMOS.items():
                if f"{{{chave}}}" in frase:
                    frase = frase.replace(f"{{{chave}}}", random.choice(opcoes))
            
            # Adiciona pequenas variações de ruído (erros comuns de digitação)
            if random.random() > 0.8:
                frase = frase.replace("a", "4").replace("e", "3") # Simula "leet speak" ou erro
                
            dataset_expandido.append((frase.lower(), categoria))
    return dataset_expandido

def treinar_e_salvar():
    print("🚀 Gerando dados sintéticos para o TCC...")
    dados = gerar_dados_em_massa(n_por_template=200) # Gera ~2800 exemplos
    
    textos, labels = zip(*dados)
    
    print(f"🧠 Treinando com {len(textos)} variações...")
    
    # Melhoramos o Vectorizer para pegar combinações de 1 e 2 palavras (n-grams)
    # Isso ajuda a entender que "som alto" é diferente de apenas "som"
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(textos)
    
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X, labels)
    
    joblib.dump(modelo, 'modelo_classificador.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("✅ Modelo robusto gerado com sucesso!")

if __name__ == "__main__":
    treinar_e_salvar()