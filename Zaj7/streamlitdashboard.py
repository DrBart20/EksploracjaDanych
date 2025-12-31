import streamlit as st
import sqlite3
import pandas as pd
from datetime import date

def init_db():
    conn = sqlite3.connect('sales.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product TEXT,
            quantity INTEGER,
            price REAL,
            date DATE,
            latitude REAL,
            longitude REAL
        )
    ''')
    conn.commit()
    return conn

conn = init_db()


st.title("Aplikacja Sprzedażowa Streamlit")


CITIES = {
    "Warszawa": (52.2297, 21.0122),
    "Kraków": (50.0647, 19.9450),
    "Wrocław": (51.1079, 17.0385),
    "Gdańsk": (54.3520, 18.6466),
    "Poznań": (52.4064, 16.9252)
}


st.sidebar.header("Dodaj nową sprzedaż")
with st.sidebar.form("sales_form"):
    product = st.text_input("Produkt")
    quantity = st.number_input("Ilość", min_value=1, step=1)
    price = st.number_input("Cena", min_value=0.0, step=0.01)
    sale_date = st.date_input("Data sprzedaży", date.today())
    

    location_mode = st.radio("Metoda lokalizacji", ["Wybierz miasto", "Wpisz ręcznie"])
    if location_mode == "Wybierz miasto":
        city_name = st.selectbox("Miasto", list(CITIES.keys()))
        lat, lon = CITIES[city_name]
    else:
        lat = st.number_input("Szerokość (lat)", value=52.0)
        lon = st.number_input("Długość (lon)", value=19.0)
        
    submitted = st.form_submit_button("Dodaj rekord")
    
    if submitted:
        query = "INSERT INTO sales (product, quantity, price, date, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?)"
        conn.execute(query, (product, quantity, price, str(sale_date), lat, lon))
        conn.commit()
        st.success("Dodano pomyślnie!")
        # st.balloons()


df = pd.read_sql_query("SELECT * FROM sales", conn)
df['date'] = pd.to_datetime(df['date'])
df['total_value'] = df['quantity'] * df['price']

st.header("Przegląd danych")


if not df.empty:
    all_products = df['product'].unique().tolist()
    selected_product = st.selectbox("Filtruj po produkcie", ["Wszystkie"] + all_products)
    
    if selected_product != "Wszystkie":
        df_filtered = df[df['product'] == selected_product]
    else:
        df_filtered = df

    if st.checkbox("Pokaż tabelę danych"):
        st.dataframe(df_filtered)

    st.header("Analiza sprzedaży")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sprzedaż dzienna (wartość)")
        daily_sales = df_filtered.groupby('date')['total_value'].sum().reset_index()
        st.line_chart(daily_sales.set_index('date'))

    with col2:
        st.subheader("Suma sprzedanych produktów wg typu")
        product_sums = df_filtered.groupby('product')['quantity'].sum().reset_index()
        st.bar_chart(product_sums.set_index('product'))

    st.header("Lokalizacje sprzedaży")
    map_filter = st.checkbox("Filtruj mapę według wybranego produktu")
    df_map = df_filtered if map_filter else df
    
    if not df_map.empty:
        st.map(df_map[['latitude', 'longitude']])
    else:
        st.info("Brak danych do wyświetlenia na mapie.")

else:
    st.info("Baza danych jest pusta. Dodaj pierwszy rekord w panelu bocznym.")

conn.close()
