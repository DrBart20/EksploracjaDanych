import streamlit as st
import pandas as pd
import numpy as np

try:
    from src import data_loader, plotting, modeling
except ImportError:
    st.error("Nie można zaimportować modułów z folderu 'src'.")
    st.stop()


st.set_page_config(
    page_title="Interaktywny Eksplorator ML",
    page_icon="",
    layout="wide"
)

st.title("Eksplorator i Predyktor ML")

if 'df' not in st.session_state:
    st.session_state['df'] = None
    st.session_state['X'] = None
    st.session_state['y'] = None
    st.session_state['model'] = None
    st.session_state['metrics'] = None
    st.session_state['y_test'] = None
    st.session_state['y_pred'] = None

st.sidebar.header("1. Źródło Danych")

uploaded_file = st.sidebar.file_uploader(
    "Wgraj plik CSV", 
    type=["csv"],
    help="Ostatnia kolumna w pliku CSV zostanie potraktowana jako zmienna docelowa (y)."
)

if uploaded_file is not None:
    df, X, y = data_loader.load_csv_data(uploaded_file)
    if df is not None:
        st.session_state['df'] = df
        st.session_state['X'] = X
        st.session_state['y'] = y

        st.session_state['model'] = None
        st.session_state['metrics'] = None
        st.sidebar.success("Własne dane wczytane!")
    else:
        st.sidebar.error("Błąd w pliku CSV..")
        
if st.sidebar.button("Wczytaj domyślny zbiór (California Housing)"):
    df, X, y = data_loader.load_california_data()
    st.session_state['df'] = df
    st.session_state['X'] = X
    st.session_state['y'] = y
    st.session_state['model'] = None
    st.session_state['metrics'] = None
    st.sidebar.success("Dane California Housing wczytane!")



if st.session_state['df'] is None:
    st.info("Proszę wczytać dane używając panelu bocznego, aby rozpocząć.")
else:
    st.success("Dane zostały pomyślnie załadowane..")
    st.dataframe(st.session_state['df'].head())

    tab_eda, tab_model, tab_predykcja = st.tabs([
        "Eksploracja Danych", 
        "Trening Modelu", 
        "Predykcja na Żywo"
    ])

    with tab_eda:
        st.header("Eksploracyjna Analiza Danych")
        
        st.subheader("Statystyki Opisowe")
        st.write(st.session_state['df'].describe())
        
        st.subheader("Mapa Korelacji")
        with st.spinner("Rysowanie mapy korelacji..."):
            fig_corr = plotting.plot_correlation_heatmap(st.session_state['df'])
            st.pyplot(fig_corr)
        
        st.subheader("Rozkład Wybranej Cechy")
        all_columns = st.session_state['df'].columns.tolist()
        col_to_plot = st.selectbox("Wybierz kolumnę do narysowania histogramu:", all_columns)
        if col_to_plot:
            with st.spinner(f"Rysowanie histogramu dla '{col_to_plot}'..."):
                fig_hist = plotting.plot_histogram(st.session_state['df'], col_to_plot)
                st.pyplot(fig_hist)
    with tab_model:
        st.header("Trening i Ewaluacja Modelu")
        
        st.sidebar.divider()
        st.sidebar.header("2. Konfiguracja Modelu")
        model_name = st.sidebar.selectbox(
            "Wybierz model regresji:",
            ("Linear Regression", "Random Forest Regressor", "SVR")
        )
        test_size = st.sidebar.slider(
            "Proporcja zbioru testowego:", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.2, 
            step=0.05
        )
        
        model_params = {}
        st.sidebar.subheader("Hiperparametry")
        if model_name == "Random Forest Regressor":
            model_params['n_estimators'] = st.sidebar.slider(
                "Liczba drzew (n_estimators):", 10, 200, 100, 10
            )
            model_params['max_depth'] = st.sidebar.slider(
                "Maksymalna głębokość (max_depth):", 2, 20, 10, 1
            )
        elif model_name == "SVR":
            model_params['C'] = st.sidebar.slider(
                "Parametr C (Regularyzacja):", 0.1, 10.0, 1.0, 0.1
            )
            model_params['kernel'] = st.sidebar.select_slider(
                "Kernel:", options=['linear', 'rbf', 'poly']
            )

        if st.sidebar.button("Trenuj Model", type="primary"):
            with st.spinner("Trening modelu w toku... To może potrwać chwilę."):

                X_train, X_test, y_train, y_test = data_loader.split_data(
                    st.session_state['X'],
                    st.session_state['y'],
                    test_size,
                    random_state=42
                )

                model_pipeline = modeling.train_model(
                    X_train, y_train, model_name, model_params
                )
                st.session_state['model'] = model_pipeline 

        
                metrics = modeling.evaluate_model(model_pipeline, X_test, y_test)
                st.session_state['metrics'] = metrics 

                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = model_pipeline.predict(X_test)
            
            st.success(f"Model {model_name} został pomyślnie wytrenowany!")

        if st.session_state['metrics']:
            st.subheader("Wyniki Ewaluacji Modelu")
            col1, col2 = st.columns(2)
            col1.metric("Współczynnik R²", f"{st.session_state['metrics']['R²']:.4f}")
            col2.metric("RMSE", f"{st.session_state['metrics']['RMSE']:.4f}")

        if st.session_state['y_pred'] is not None:
            st.subheader("Wykres Rzeczywiste vs Przewidywane")
            fig_pred = plotting.plot_predictions(
                st.session_state['y_test'],
                st.session_state['y_pred']
            )
            st.pyplot(fig_pred)

    with tab_predykcja:
        st.header("Predykcja na Żywo")
        
        if st.session_state['model'] is None:
            st.warning("Najpierw musisz wytrenować model w zakładce.")
        else:
            st.info("Użyj suwaków i pól w panelu bocznym, aby wprowadzić własne dane do predykcji.")
            
            st.sidebar.divider()
            st.sidebar.header("3. Parametry do Predykcji")
            
            input_data = {}
            for col in st.session_state['X'].columns:
                min_val = float(st.session_state['X'][col].min())
                max_val = float(st.session_state['X'][col].max())
                mean_val = float(st.session_state['X'][col].mean())
                
                input_data[col] = st.sidebar.number_input(
                    label=f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100  
                )
            
            input_df = pd.DataFrame([input_data])
            st.subheader("Wybrane parametry wejściowe:")
            st.dataframe(input_df)
            
            prediction = modeling.make_prediction(
                st.session_state['model'],
                input_df
            )
            
            st.subheader(f"Przewidywana wartość:")
            st.markdown(f"## `{prediction[0]:.4f}`")
            st.markdown(f"Wartość przewidywana: **${prediction[0] * 100_000:,.2f}**")
            
        

