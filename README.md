# 🤖 AI Option Pricer & Deep Hedging (From Scratch)

Plateforme d'intelligence artificielle et de finance quantitative dédiée au pricing d'options et à la couverture dynamique des risques (*Deep Hedging*). Ce projet s'affranchit des boîtes noires de Deep Learning pour coder l'intégralité du réseau de neurones et des algorithmes d'optimisation en pur Python/NumPy (*From Scratch*).

## 📊 Fonctionnalités Clés

1. **Modèle de Référence (Black-Scholes Classique) :** 
   - Implémentation analytique complète des prix et sensibilités (Delta) pour les options Calls et Puts.
   - Visualisation interactive de l'impact du strike et des paramètres de marché.

2. **Réseau de Neurones 100% "From Scratch" (IA vs Historique) :**
   - Extraction des données de marché historiques (`yfinance`) et simulation de frictions réalistes (*bruit de sentiment*).
   - Architecture de Deep Learning personnalisée : couches cachées (ReLU), couche de sortie (Softplus pour garantir des prix strictement positifs) et optimiseur Adam codé à la main.
   - Calcul d'un **Deep Delta** robuste par la méthode numérique des différences centrales, capable de s'adapter aux asymétries et aux chocs de marché que le modèle classique ignore.

3. **Prédiction Future & Stress-Test (Monte Carlo) :**
   - Génération de trajectoires stochastiques futures pour le sous-jacent via un Mouvement Brownien Géométrique.
   - Évaluation dynamique par l'IA de l'évolution des prix et de la couverture (*Delta*) à travers de multiples scénarios de crise (*multivers financiers*).

---

## 🛠️ Installation et Lancement

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/votre-nom-d-utilisateur/ai-option-pricer-deep-hedging.git](https://github.com/votre-nom-d-utilisateur/ai-option-pricer-deep-hedging.git)
   cd ai-option-pricer-deep-hedging
2. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
3. **Lancer le frontend :**
   ```bash
   streamlit run dashboardv2.py
