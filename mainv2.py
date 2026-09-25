# Code pour la partie backend du projet IA

# Implémentation des librairies nécessaires pour le projet d'IA de pricing d'options
import numpy as np 
import pandas as pd
from scipy.stats import norm
import yfinance as yf # Pour récupérer les données historiques de l'action et des options
from sklearn.preprocessing import StandardScaler # Pour normaliser les données d'entrée du réseau de neurones
from datetime import datetime # Pour manipuler les dates et calculer les maturités des options

# Classe pour le modèle de Black-Scholes, qui servira de référence théorique pour les prix et deltas des options Calls et Puts.
class BlackScholesModel:

    # Calcul de d1 et d2, les paramètres clés du modèle de Black-Scholes, avec des protections contre les divisions par zéro.
    @staticmethod
    def d1_d2(S0, K, T, r, sigma):
        eps = 1e-12
        sigma = max(sigma, eps)
        T = max(T, eps)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    # Calcul du prix d'une option Call selon le modèle de Black-Scholes.
    @classmethod
    def call_price(cls, S0, K, T, r, sigma):
        d1, d2 = cls.d1_d2(S0, K, T, r, sigma)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    # Calcul du Delta d'une option Call selon le modèle de Black-Scholes.
    @classmethod
    def call_delta(cls, S0, K, T, r, sigma):
        d1, _ = cls.d1_d2(S0, K, T, r, sigma)
        return norm.cdf(d1)

    # Calcul du prix d'une option Put selon le modèle de Black-Scholes.
    @classmethod
    def put_price(cls, S0, K, T, r, sigma):
        d1, d2 = cls.d1_d2(S0, K, T, r, sigma)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    # Calcul du Delta d'une option Put selon le modèle de Black-Scholes.
    @classmethod
    def put_delta(cls, S0, K, T, r, sigma):
        d1, _ = cls.d1_d2(S0, K, T, r, sigma)
        return norm.cdf(d1) - 1.0

# Classe pour récupérer les données de marché historiques, en simulant des prix d'options réalistes basés 
# sur les prix du sous-jacent et les paramètres de marché.
class MarketDataFetcher:

    # Récupère les données historiques du sous-jacent et génère des prix d'options simulés avec des frictions de marché réalistes.  
    @staticmethod
    def get_historical_options(ticker_symbol, option_type='Both'):
        ticker = yf.Ticker(ticker_symbol)
        hist_underlying = ticker.history(period="1mo")
        if hist_underlying.empty:
            return None

        dates = hist_underlying.index
        spot_prices = hist_underlying['Close'].values
        
        data = []
        # Le strike est fixé au dernier prix du sous-jacent pour simuler une option at-the-money, 
        # ce qui est courant pour les données d'entraînement.
        strike = spot_prices[-1] 
        expiration_days = 30
        
        for i, date in enumerate(dates):
            S0 = spot_prices[i]
            # La maturité T est calculée en fonction du nombre de jours restants jusqu'à l'expiration, 
            # avec une valeur minimale pour éviter les divisions par zéro
            T = max((expiration_days - (len(dates) - 1 - i)) / 365.0, 0.005)
            r = 0.05
            iv = 0.25 
            
            # Le sentiment de marché est simulé pour introduire des biais réalistes dans les prix d'options,
            sentiment = (S0 / strike - 1) * 0.15 
            # Le bruit aléatoire simule les frictions de marché, les erreurs de modélisation et 
            # les comportements irrationnels des traders, rendant les données plus réalistes pour l'entraînement du modèle d'IA.
            noise = 1 + np.random.normal(0, 0.03)

            if option_type in ['Call', 'Both']:
                bs_price = BlackScholesModel.call_price(S0, strike, T, r, iv)
                bs_delta = BlackScholesModel.call_delta(S0, strike, T, r, iv)
                # Le prix réel intègre des frictions et biais que BS ignore
                real_price = max(0.01, (bs_price + max(0, sentiment)) * noise)
                data.append([date, S0, strike, T, r, iv, 1, real_price, bs_price, bs_delta])
                
            if option_type in ['Put', 'Both']:
                bs_price = BlackScholesModel.put_price(S0, strike, T, r, iv)
                bs_delta = BlackScholesModel.put_delta(S0, strike, T, r, iv)
                # Le prix réel intègre des frictions et biais que BS ignore
                real_price = max(0.01, (bs_price - min(0, sentiment)) * noise)
                data.append([date, S0, strike, T, r, iv, 0, real_price, bs_price, bs_delta])
                
        df = pd.DataFrame(data, columns=['Date', 'S0', 'strike', 'T', 'r', 'impliedVolatility', 'is_call', 'lastPrice', 'BS_Price', 'BS_Delta'])
        df.set_index('Date', inplace=True)
        return df

# Classe pour normaliser les données d'entrée du réseau de neurones, en implémentant manuellement 
# un StandardScaler pour éviter les dépendances externes et mieux comprendre le processus de normalisation.
class StandardScalerFromScratch:

    # Fonction d'initialisation qui prépare les variables pour stocker la moyenne et l'écart-type 
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    # Fonction pour calculer la moyenne et l'écart-type des données d'entraînement, et normaliser les données
    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # On ajoute un epsilon pour éviter la division par zéro
        self.std_[self.std_ == 0] = 1e-8 
        return (X - self.mean_) / self.std_

    # Fonction pour normaliser de nouvelles données 
    def transform(self, X):
        return (X - self.mean_) / self.std_

# Fonction d'activation basique style Relu 
def relu(Z):
    return np.maximum(0, Z)

# Dérivée de cette fonction d'activation 
def relu_deriv(Z):
    return (Z > 0).astype(float)

# Fonction d'activation Softplus, donnant du positif et une dérivée lisse 
def softplus(Z):
    return np.log1p(np.exp(-np.abs(Z))) + np.maximum(Z, 0)

# Dérivée de la fonction d'activation Softplus, qui est la fonction sigmoïde
def softplus_deriv(Z):
    return 1.0 / (1.0 + np.exp(-np.clip(Z, -250, 250)))

# Classe pour le réseau de neurones personnalisé, qui implémente une architecture simple avec des fonctions d'activation ReLU et Softplus,
class CustomOptionNet:

    # Fonction d'initialisation qui crée les poids et biais du réseau de neurones
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.params = {}
        self.grads = {}
        
        # Initialisation de He (Xavier) pour les poids
        for i in range(1, self.num_layers):
            self.params[f'W{i}'] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1])
            self.params[f'b{i}'] = np.zeros((1, layer_sizes[i]))
            
        # Paramètres pour l'optimiseur Adam
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    # Fonction pour effectuer la propagation avant à travers le réseau de neurones, 
    # en stockant les activations et les pré-activations pour la rétropropagation
    def forward(self, X):
        self.cache = {'A0': X}
        A = X
        L = self.num_layers - 1
        
        # Couches cachées (ReLU)
        for i in range(1, self.num_layers - 1):
            # Z est la pré-activation, A est l'activation après ReLU
            Z = np.dot(A, self.params[f'W{i}']) + self.params[f'b{i}']
            A = relu(Z)
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = A
            
        # Couche de sortie (Softplus)
        # L est l'indice de la dernière couche, qui utilise une activation Softplus
        Z = np.dot(A, self.params[f'W{L}']) + self.params[f'b{L}']
        A = softplus(Z)
        self.cache[f'Z{L}'] = Z
        self.cache[f'A{L}'] = A
        
        return A

    # Fonction pour effectuer la rétropropagation à travers le réseau de neurones, en calculant les gradients des poids et biais
    def backward(self, Y):
        m = Y.shape[0]
        L = self.num_layers - 1
        
        # Dérivée de la loss MSE : dLoss/dA = 2 * (A - Y) / m
        dA = 2.0 * (self.cache[f'A{L}'] - Y) / m
        
        # Rétropropagation couche de sortie (Softplus)
        dZ = dA * softplus_deriv(self.cache[f'Z{L}'])
        self.grads[f'dW{L}'] = np.dot(self.cache[f'A{L-1}'].T, dZ)
        self.grads[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True)
        
        # Rétropropagation couches cachées (ReLU)
        for i in reversed(range(1, L)):
            dA = np.dot(dZ, self.params[f'W{i+1}'].T)
            dZ = dA * relu_deriv(self.cache[f'Z{i}'])
            self.grads[f'dW{i}'] = np.dot(self.cache[f'A{i-1}'].T, dZ)
            self.grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True)

    # Fonction pour mettre à jour les poids et biais du réseau de neurones en utilisant l'optimiseur Adam, 
    # qui adapte les taux d'apprentissage pour chaque paramètre en fonction des moments passés des gradients.
    def step_adam(self, learning_rate):
        self.t += 1
        for key in self.params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * self.grads['d' + key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (self.grads['d' + key]**2)
            
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            
            self.params[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Classe pour le pricer d'options basé sur l'IA, qui intègre la préparation des données, 
# l'entraînement du modèle de réseau de neurones personnalisé,
class AIOptionPricerFromScratch:

    # Fonction d'initialisation qui crée une instance du StandardScaler personnalisé et du réseau de neurones personnalisé,
    def __init__(self):
        self.scaler_X = StandardScalerFromScratch()
        # Architecture : 6 entrées -> 64 -> 64 -> 32 -> 1 sortie
        self.model = CustomOptionNet(layer_sizes=[6, 64, 64, 32, 1])

    # Fonction pour préparer les données d'entrée du réseau de neurones
    def prepare_data(self, df):
        X = df[['S0', 'strike', 'T', 'r', 'impliedVolatility', 'is_call']].values
        y = df['lastPrice'].values.reshape(-1, 1)
        return X, y

    # Fonction pour entraîner le modèle de réseau de neurones sur les données d'entraînement, en effectuant des passes avant et arrière
    def train(self, df, epochs=600, lr=0.01):
        X, Y = self.prepare_data(df)
        X_scaled = self.scaler_X.fit_transform(X)
        
        final_loss = 0
        for epoch in range(epochs):
            # Forward pass
            predictions = self.model.forward(X_scaled)
            
            # Calcul de la Loss (MSE)
            loss = np.mean(np.square(predictions - Y))
            final_loss = loss
            
            # Backward pass et mise à jour
            self.model.backward(Y)
            self.model.step_adam(learning_rate=lr)
            
        return final_loss

    # Fonction pour faire des prédictions de prix d'options à partir de nouvelles données, en utilisant le modèle entraîné
    def predict(self, df):
        X, _ = self.prepare_data(df)
        X_scaled = self.scaler_X.transform(X)
        preds = self.model.forward(X_scaled)
        return preds.flatten()

    # Fonction pour calculer le Deep Delta, qui est la sensibilité du prix de l'option par rapport au prix du sous-jacent,
    def compute_deep_delta(self, df, epsilon=1e-4):
        X, _ = self.prepare_data(df)
        
        X_up = X.copy()
        X_down = X.copy()
        
        # On modifie seulement S0 (index 0)
        X_up[:, 0] += epsilon
        X_down[:, 0] -= epsilon
        
        X_up_scaled = self.scaler_X.transform(X_up)
        X_down_scaled = self.scaler_X.transform(X_down)
        
        preds_up = self.model.forward(X_up_scaled)
        preds_down = self.model.forward(X_down_scaled)
        
        # Calcul du delta numérique : (f(S0 + epsilon) - f(S0 - epsilon)) / (2 * epsilon)
        deep_delta = (preds_up - preds_down) / (2 * epsilon)
        return deep_delta.flatten()

# Classe pour simuler des scénarios futurs de prix du sous-jacent, en utilisant un modèle de Mouvement Brownien Géométrique,
class FutureSimulator:

    # La méthode génère n scénarios d'évolution du prix du sous-jacent pour les prochains jours, 
    # en utilisant la formule du Mouvement Brownien Géométrique,  
    @staticmethod
    def generate_future_paths(S0, mu, sigma, days_ahead, n_scenarios=5):
        dt = 1 / 365.0
        paths = np.zeros((days_ahead, n_scenarios))
        paths[0] = S0
        
        for t in range(1, days_ahead):
            # Z est un choc aléatoire (loi normale)
            Z = np.random.standard_normal(n_scenarios)
            # Formule du Mouvement Brownien Géométrique
            paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            
        return paths