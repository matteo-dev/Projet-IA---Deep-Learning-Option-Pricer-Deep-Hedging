# Code pour la partie frontend du projet IA

# Importation des bibliothèques nécessaires
import streamlit as st # Framework de développement d'applications web interactives
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from mainv2 import BlackScholesModel, MarketDataFetcher, AIOptionPricerFromScratch, FutureSimulator # Importation des classes du projet

st.set_page_config(page_title="Projet IA", layout="wide")

# Interface utilisateur avec Streamlit, permettant de naviguer entre les différentes fonctionnalités du projet
st.sidebar.title("Navigation Projet IA")
page = st.sidebar.radio("Page", [
    "1. Modèle B&S Classique", 
    "2. IA vs Historique (1 Mois)", 
    "3. Prédiction Future (Monte Carlo)" 
])

# Page 1 : Calculateur Black-Scholes Théorique
if page == "1. Modèle B&S Classique":
    st.title("Calculateur Black-Scholes Théorique")

    # Sélection du type d'option à analyser  
    col_type, _ = st.columns(2)
    with col_type:
        opt_type = st.radio("Type d'Option", ["Call", "Put", "Both (Les Deux)"], horizontal=True)
    
    # Saisie des paramètres d'entrée pour le modèle Black-Scholes
    col1, col2 = st.columns([1, 2])
    with col1:
        S0 = st.number_input("Prix de l'actif (S0)", value=100.0)
        K = st.number_input("Strike (K)", value=100.0)
        T = st.number_input("Maturité en années (T)", value=1.0)
        r = st.number_input("Taux sans risque (r)", value=0.05)
        sigma = st.number_input("Volatilité (sigma)", value=0.2)
        
        # Affichage des résultats théoriques
        if opt_type in ["Call", "Both (Les Deux)"]:
            price_c = BlackScholesModel.call_price(S0, K, T, r, sigma)
            delta_c = BlackScholesModel.call_delta(S0, K, T, r, sigma)
            st.success(f"**Prix théorique (Call) : {price_c:.4f} €**")
            st.info(f"**Delta théorique (Call) : {delta_c:.4f}**")

        if opt_type in ["Put", "Both (Les Deux)"]:
            price_p = BlackScholesModel.put_price(S0, K, T, r, sigma)
            delta_p = BlackScholesModel.put_delta(S0, K, T, r, sigma)
            st.error(f"**Prix théorique (Put) : {price_p:.4f} €**")
            st.warning(f"**Delta théorique (Put) : {delta_p:.4f}**")

    # Graphique du prix en fonction du strike 
    with col2:
        strikes = np.linspace(S0*0.5, S0*1.5, 50)
        fig = go.Figure()
        
        if opt_type in ["Call", "Both (Les Deux)"]:
            prices_c = [BlackScholesModel.call_price(S0, k, T, r, sigma) for k in strikes]
            fig.add_trace(go.Scatter(x=strikes, y=prices_c, mode='lines', name='Call Price', line=dict(color='blue')))

        if opt_type in ["Put", "Both (Les Deux)"]:
            prices_p = [BlackScholesModel.put_price(S0, k, T, r, sigma) for k in strikes]
            fig.add_trace(go.Scatter(x=strikes, y=prices_p, mode='lines', name='Put Price', line=dict(color='orange')))
            
        titre_graph = "Prix en fonction du Strike (K)"
        fig.update_layout(title=titre_graph, xaxis_title="Strike (K)", yaxis_title="Prix")
        st.plotly_chart(fig, use_container_width=True)

# Page 2 : Comparaison entre les prédictions de l'IA et les données historiques
elif page == "2. IA vs Historique (1 Mois)":
    st.title("Analyse Temporelle : Deep Learning vs Marché")
    
    # MISE À JOUR TEXTUELLE : On retire PyTorch et on valorise le code "From Scratch" et la différence centrale
    with st.expander("💡 Pourquoi l'IA est supérieure à Black-Scholes ? (Explication)"):
        st.markdown("""
        - **Bruit et Sentiment de marché :** Le modèle Black-Scholes est figé et suppose des marchés parfaits (continus, sans frictions). L'IA est capable d'apprendre la "prime de risque" (la peur ou l'euphorie du marché) qui dévie les prix réels de la théorie.
        - **Réseau de Neurones 100% "From Scratch" :** Au lieu d'utiliser des boîtes noires comme PyTorch, ce modèle a été entièrement codé à la main en Python/NumPy (Forward pass, Rétropropagation, Optimiseur Adam).
        - **Un Delta Dynamique (Différence Centrale) :** Le Delta théorique (B&S) est aveugle aux chocs soudains. Notre IA calcule un **Deep Delta** via la méthode numérique des différences centrales. Il s'adapte à la volatilité asymétrique historique, offrant une bien meilleure protection au trader.
        """)

    col_t, col_s = st.columns(2)
    with col_t:
        ticker = st.text_input("Ticker (ex: AAPL, SPY, CAC40=X)", value="AAPL")
    with col_s:
        target_opt = st.radio("Analyser :", ["Call", "Put", "Both (Les Deux)"], horizontal=True)
    
    opt_map = {"Call": "Call", "Put": "Put", "Both (Les Deux)": "Both"}
    
    if st.button("Télécharger l'historique et Entraîner l'IA"):
        # MISE À JOUR TEXTUELLE
        with st.spinner(f"Extraction du mois historique pour {ticker} et entraînement du réseau de neurones From Scratch..."):
            df_hist = MarketDataFetcher.get_historical_options(ticker, option_type=opt_map[target_opt])

        if df_hist is not None and not df_hist.empty:
            ai_model = AIOptionPricerFromScratch()
            loss = ai_model.train(df_hist, epochs=600)
            
            df_hist['AI_Price'] = ai_model.predict(df_hist)
            df_hist['AI_Delta'] = ai_model.compute_deep_delta(df_hist)
            
            st.success(f"Entraînement réussi ! L'IA a appris les inefficacités du marché sur le dernier mois (MSE Loss: {loss:.4f}).")
            
            def plot_results(df_subset, name_suffix):
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(x=df_subset.index, y=df_subset['lastPrice'], mode='markers', name='Prix Réel Marché', marker=dict(color='white')))
                fig_p.add_trace(go.Scatter(x=df_subset.index, y=df_subset['BS_Price'], mode='lines', name='B&S Théorique', line=dict(color='red')))
                fig_p.add_trace(go.Scatter(x=df_subset.index, y=df_subset['AI_Price'], mode='lines', name='Prédiction IA', line=dict(color='green', dash='dash')))
                fig_p.update_layout(title=f"Evolution du Prix ({name_suffix})", xaxis_title="Date", yaxis_title="Prix")
                
                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(x=df_subset.index, y=df_subset['BS_Delta'], mode='lines', name='Delta B&S', line=dict(color='red')))
                fig_d.add_trace(go.Scatter(x=df_subset.index, y=df_subset['AI_Delta'], mode='lines', name='Deep Delta (IA)', line=dict(color='purple', dash='dot')))
                fig_d.update_layout(title=f"Couverture (Delta {name_suffix})", xaxis_title="Date", yaxis_title="Valeur du Delta")
                
                c1, c2 = st.columns(2)
                c1.plotly_chart(fig_p, use_container_width=True)
                c2.plotly_chart(fig_d, use_container_width=True)

            if target_opt in ["Call", "Both (Les Deux)"]:
                st.subheader("Analyse de l'Option CALL")
                plot_results(df_hist[df_hist['is_call'] == 1], "Call")
  
            if target_opt in ["Put", "Both (Les Deux)"]:
                st.subheader("Analyse de l'Option PUT")
                plot_results(df_hist[df_hist['is_call'] == 0], "Put")
        else:
            st.error("Impossible de récupérer les données pour ce ticker.")

# Page 3 : Simulation de scénarios futurs (Monte Carlo)
elif page == "3. Prédiction Future (Monte Carlo)":
    st.title("Prédiction : Stress-Test du Portefeuille")
    st.markdown("""
    **Concept :** Nous ne pouvons pas prédire l'avenir avec certitude. Cependant, nous pouvons simuler plusieurs futurs possibles (Scénarios Monte Carlo) et demander à l'IA comment elle réagirait (Prix et Couverture) dans chacun de ces futurs.
    """)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        ticker = st.text_input("Ticker (ex: AAPL)", value="AAPL", key="ticker_pred")
        days_ahead = st.slider("Jours à prédire", min_value=5, max_value=30, value=10)
    with col_p2:
        target_opt = st.radio("Type d'option :", ["Call", "Put", "Both (Les Deux)"], horizontal=True, key="opt_pred")
        
    opt_map_pred = {"Call": "Call", "Put": "Put", "Both (Les Deux)": "Both"}
        
    if st.button("Simuler l'Avenir"):
        with st.spinner("Apprentissage de l'historique et simulation des futurs..."):
            
            df_hist = MarketDataFetcher.get_historical_options(ticker, option_type=opt_map_pred[target_opt])
            if df_hist is not None and not df_hist.empty:
                ai_model = AIOptionPricerFromScratch()
                ai_model.train(df_hist, epochs=500)
                
                last_S0 = df_hist['S0'].iloc[-1]
                strike = df_hist['strike'].iloc[-1]
                r = 0.05
                iv = df_hist['impliedVolatility'].iloc[-1]
                
                future_paths = FutureSimulator.generate_future_paths(last_S0, mu=0.08, sigma=iv, days_ahead=days_ahead, n_scenarios=3)
                
                def plot_future_simulations(is_call_val, name_suffix):
                    fig_fut_price = go.Figure()
                    fig_fut_delta = go.Figure()
                    dates_future = [f"J+{i}" for i in range(days_ahead)]
                    colors = ['blue', 'orange', 'red']
                    
                    for i in range(3):
                        path_S0 = future_paths[:, i]
                        
                        df_future = pd.DataFrame({
                            'S0': path_S0,
                            'strike': strike,
                            'T': [max((30 - t)/365.0, 0.005) for t in range(days_ahead)],
                            'r': r,
                            'impliedVolatility': iv,
                            'is_call': is_call_val,
                            'lastPrice': 0 
                        })
                        
                        pred_prices = ai_model.predict(df_future)
                        pred_deltas = ai_model.compute_deep_delta(df_future)
                        
                        fig_fut_price.add_trace(go.Scatter(x=dates_future, y=pred_prices, mode='lines+markers', name=f'Scénario {i+1} (Prix)', line=dict(color=colors[i])))
                        fig_fut_delta.add_trace(go.Scatter(x=dates_future, y=pred_deltas, mode='lines+markers', name=f'Scénario {i+1} (Delta)', line=dict(color=colors[i], dash='dot')))

                    fig_fut_price.update_layout(title=f"Prédiction IA : Prix {name_suffix}", xaxis_title="Jours futurs", yaxis_title="Prix IA (€)")
                    fig_fut_delta.update_layout(title=f"Prédiction IA : Deep Delta {name_suffix}", xaxis_title="Jours futurs", yaxis_title="Delta IA")
                    
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(fig_fut_price, use_container_width=True)
                    c2.plotly_chart(fig_fut_delta, use_container_width=True)

                st.success("Simulations Monte Carlo terminées et évaluées par l'IA !")
                
                if target_opt in ["Call", "Both (Les Deux)"]:
                    st.subheader("Projection pour l'Option CALL")
                    plot_future_simulations(1, "Call")

                if target_opt in ["Put", "Both (Les Deux)"]:
                    st.subheader("Projection pour l'Option PUT")
                    plot_future_simulations(0, "Put")
                
            else:
                st.error("Erreur lors de la récupération des données.")