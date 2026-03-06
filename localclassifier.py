import joblib

class LocalClassifier:
    """
    Carrega o modelo de Machine Learning treinado localmente
    para classificar a intenção do morador sem custos de API.
    """
    def __init__(self):
        try:
            # Carrega os arquivos gerados pelo seu script 'treinador.py'
            self.modelo = joblib.load('modelo_classificador.pkl')
            self.vectorizer = joblib.load('vectorizer.pkl')
            print("✅ [Brain] Modelo de classificação local carregado com sucesso.")
        except FileNotFoundError:
            self.modelo = None
            self.vectorizer = None
            print("⚠️ [Brain] Aviso: Arquivos .pkl não encontrados. O sistema não registrará categorias até você rodar o treinador.py.")

    def classificar(self, texto):
        """
        Transforma o texto em vetor e prediz a categoria.
        """
        if self.modelo is None or self.vectorizer is None:
            return "Indefinido"
        
        try:
            # Transforma a entrada do morador usando o mesmo vocabulário do treino
            vetor = self.vectorizer.transform([texto.lower()])
            # Prediz a categoria (Barulho, Obras, etc.)
            categoria = self.modelo.predict(vetor)[0]
            return categoria
        except Exception as e:
            print(f"Erro na classificação local: {e}")
            return "Erro na Classificação"