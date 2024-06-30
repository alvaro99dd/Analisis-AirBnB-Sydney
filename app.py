import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium
import geopandas as gpd
from streamlit.components.v1 import html

st.set_page_config(page_title="Airbnb Sydney", page_icon=":house:",layout="wide") #configuración de la página

#Cargar datos
listings = pd.read_csv("https://raw.githubusercontent.com/alvaro99dd/Analisis-AirBnB-Sydney/main/Recursos/listings_clean.zip")

#Funciones
#Función para limpiar los outliers
def clean_outliers(df_aux, column: str):
    Q1 = df_aux[column].quantile(0.25)
    Q3 = df_aux[column].quantile(0.75)
    IQR = Q3 - Q1
    df_aux = df_aux[(df_aux[column] >= Q1-1.5*IQR) & (df_aux[column] <= Q3 + 1.5*IQR)]
    return df_aux

def change_page(seleccion):
    """Muestra la página en función de lo que seleccione el usuario en el sidebar"""
    match seleccion:
        case "Inicio":
            inicio()
        case "Datos usados":
            datos_usados()
        case "Importancia del precio":
            precio()
        case "Importancia del vecindario":
            vecindario()
        case "Importancia del rating":
            rating()

st.title("Análisis exploratorio de Airbnbs en Sydney")
st.sidebar.title("Opciones de la tabla")
pestaña = st.sidebar.radio("Selecciona una pestaña:", ("Inicio", "Datos usados", "Importancia del Precio", "Importancia del Vecindario", "Importancia del rating"))

change_page(pestaña)

def inicio():
    st.subheader("Investigación exhaustiva para decidir en qué propiedades y barrios es más rentable invertir")
    cols = st.columns(2)
    with cols[0]:
        st.image("https://content.r9cdn.net/rimg/dimg/12/98/b1e36771-city-2258-163f4d7f814.jpg?crop=true&width=1020&height=498")
        st.caption("Fuente: Kayak.es")
    with cols[1]:
        with st.expander("Resumen análisis"):
            st.write('''
            A lo largo de esta aplicación observaremos
            las diferentes variables que afectan no sólo al precio
            medio de la vivienda sino también a sus puntuaciones en la aplicación
            AirBnB, sirviendo como guía o consejos para nuestros clientes
            a la hora de invertir en la zona de Sydney.
            ''')
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.image("https://www.eleconomista.es/finanzas-personales/wp-content/uploads/2023/12/Untitled-1-1.png", width=300)
            # st.markdown("##### Datos analizados")
            # st.markdown("##### Precio medio")
            # st.markdown("##### Vecindarios")
            # st.markdown("##### Rating")

def datos_usados():
    tabsInicio = st.tabs(["Datos Cargados"])
    with tabsInicio[0]:
        filtrotabla = st.checkbox("Mostrar datos analizados", value=False)
        if filtrotabla:
            df = pd.DataFrame(data=listings, columns=['neighbourhood', 'price_eur', 'room_type', 'property_type',
            'number_of_reviews', 'review_scores_rating', 'amenities', 'accommodates', 'review_scores_location', 'bedrooms'])
            st.subheader("Datos Analizados")
            st.dataframe(df)
        else:
            
            st.subheader("Datos Preprocesados")
            st.dataframe(listings)

def precio():
    tabsPrecio = st.tabs(["Según propiedad", "Según valoraciones", "Según comodidades", "Según barrio"])
    with tabsPrecio[0]:
        cols = st.columns(2)
        with cols[0]:
            st.markdown("#####")
            # Grafica precio medio por tipo de propiedad
            mean_price = listings.groupby('property_type')['price_eur'].mean().round(2).sort_values(ascending=False).head(10)
            colors = ['#FF3131' if i < 1 else '#36454F' for i in range(len(mean_price))]

            fig = px.bar(mean_price, x=mean_price.index, y=mean_price.values, labels={'y':'Precio Medio', 'property_type':'Tipo de Propiedad'})
            fig.update_layout(
                xaxis_title='Tipo de Propiedad',
                yaxis_title='Precio Medio',
                title='Precio Medio por Tipo de Propiedad'
            )
            fig.update_traces(marker_color=colors)
            st.plotly_chart(fig, use_container_width=True)
        with cols[1]:
            # Grafica de barras para mostrar el precio medio por número de habitaciones con el top 5 tipos de propiedad
            df_aux = listings.copy()
            grouped_df = df_aux.groupby(['bedrooms', 'property_type']).agg(
            price_eur_mean=('price_eur', 'mean')
            ).reset_index()

            top_5_property_types = df_aux['property_type'].value_counts().nlargest().index

            filtered_grouped_df = grouped_df[grouped_df['property_type'].isin(top_5_property_types)]

            final_filtered_df = filtered_grouped_df[(filtered_grouped_df['bedrooms'] >= 1)&(filtered_grouped_df['bedrooms'] <= 10)]
            fig = px.bar(final_filtered_df,
            x="bedrooms",
            y="price_eur_mean",
            color="property_type",
            barmode="group",
            title="Precio Medio(€) por Número de Habitaciones con el Top 5 Tipos de Propiedad",
            labels={"bedrooms": "Número de Habitaciones", "price_eur_mean": "Precio Medio (EUR)", "property_type": "Tipo de propiedad"},
            category_orders={"bedrooms": sorted(final_filtered_df['bedrooms'].unique())})  # Ordenar las categorías de 'bedrooms'

            fig.update_layout(
            plot_bgcolor="white",
            yaxis=dict(title='Precio Medio (EUR)', gridcolor='lightgrey'),
            xaxis=dict(title='Número de Habitaciones'),
            legend=dict(title='Tipo de Propiedad')
            )
            st.plotly_chart(fig, use_container_width=True)
    with tabsPrecio[1]:
        # Mapa de propiedades por Precio que tengan mas de 10 Reseñas
        df_filtered = listings.copy()
        df_filtered['review_scores_rating'] = listings[listings["number_of_reviews"] > 10]["review_scores_rating"]

        df_filtered = df_filtered.dropna(subset=["review_scores_rating"])
        df_filtered = df_filtered.sort_values(by="review_scores_rating", ascending=True)
        df_filtered['adjusted_size'] = df_filtered['price_eur'] + 35

        fig = px.scatter_mapbox(df_filtered, lat="latitude", lon="longitude",
        color="price_eur", size="adjusted_size",
        size_max=55, 
        animation_frame="review_scores_rating",
        zoom=9, mapbox_style="open-street-map",
        color_continuous_scale="viridis",
        title="Propiedades por Precio que tengan mas de 10 Reseñas", range_color=[0, 500],
        labels={"price_eur": "Precio (EUR)", "review_scores_rating": "Valoración Total"},
        height=800)
        st.plotly_chart(fig, use_container_width=True)

        # Grafica districución todas las reviews, valoración total y precio
        fig = px.scatter(listings, x='review_scores_rating', y='number_of_reviews', color="price_eur",
        size="number_of_reviews", range_color=[0, 500], range_x=[4, 5],
        title="Distribución todas las Valoraciones, Valoración Total y Precio",
        color_continuous_scale="Portland",
        labels={"review_scores_rating": "Valoración Total", "number_of_reviews": "Total Reviews", "price_eur": "Precio(€)"},)
        st.plotly_chart(fig, use_container_width=True)
    with tabsPrecio[2]:
        #Grafica total de comodidades por numero de propiedades
        df_aux = listings.copy()
        df_aux['total_amenities'] = df_aux['amenities'].apply(lambda x: len(x.split(',')) if x != '[]' else 0)

        mean_price_amenities=df_aux.groupby('total_amenities')['price_eur'].mean().round(2).reset_index(name='mean_price')
        count_properties_amenities = df_aux.groupby('total_amenities')['price_eur'].count().reset_index(name='count_properties')

        result_df = pd.merge(mean_price_amenities, count_properties_amenities, on='total_amenities')

        fig = px.scatter(result_df, x='total_amenities', y='count_properties', color="mean_price", size="count_properties", title='Numero de propiedades por total de comodidades y precio medio', range_color=[0, 500],
                        labels={'mean_price': 'Precio Medio(€)', 'count_properties': 'Numero de propiedades', 'total_amenities': 'Total de comodidades'}, color_continuous_scale='Plasma')
        fig.update_layout(
            xaxis_title='Total de comodidades',
            yaxis_title='Numero de propiedades',
            title='Total de comodidades por numero de propiedades',
        )
        st.plotly_chart(fig, use_container_width=True)
    with tabsPrecio[3]:
        listings_clean_df = listings.copy()
        listings_clean_df.dropna(subset=['neighbourhood', 'price_eur'], inplace=True)
        price_mean_by_neighbourhood = listings_clean_df.groupby('neighbourhood', as_index=False)['price_eur'].mean().round(2)
        price_mean_by_neighbourhood.rename(columns={'price_eur': 'price_mean_eur'}, inplace=True)

        fig = px.treemap(price_mean_by_neighbourhood, 
        path=['neighbourhood'], 
        values='price_mean_eur',
        color='price_mean_eur', 
        color_continuous_scale='RdYlGn',
        title='Distribución de Precio Medio por Barrio',
        labels={'price_mean_eur': 'Precio Medio (€)'})
        fig.update_traces(hovertemplate='Barrio: %{label}<br>Precio Medio (€): %{value}')
        st.plotly_chart(fig, use_container_width=True)

def vecindario():
    tabsVecindario = st.tabs(["Propiedades según vecindario", "Distribución habitaciones", "Disponibilidad", "Según puntuación de ubicación"])
    with tabsVecindario[0]:
        cols = st.columns(2)
        with cols[0]:
            st.markdown("<br/>", unsafe_allow_html=True)
            # Gráfico de barras para mostrar el top 10 de vecindarios con más propiedades
            feq=listings['neighbourhood'].value_counts(ascending=False).head(10) # Calculamos el top 10 de vecindarios con más propiedades
            colors = ['#FF3131' if i < 4 else '#36454F' for i in range(len(feq))] # Destacamos el top 4
            # Creamos y mostramos el gráfico
            fig = feq.plot.barh(figsize=(10, 5), color=colors, width=1, subplots=True)
            plt.title("Top 10 vecindarios con más propiedades", fontsize=20)
            plt.xlabel('Número de propiedades', fontsize=12)
            plt.ylabel('', fontsize=12)

            st.pyplot(plt)
        with cols[1]:
            #Tipo de propiedades según el top 4 de vecindarios
            # Creamos un dataframe auxiliar con el top 4 visto anteriormente
            df_aux = listings[(listings['neighbourhood'] == "Sydney") | (listings['neighbourhood'] == "Waverley") | (listings['neighbourhood'] == "Pittwater") | (listings['neighbourhood'] == "Randwick")]
            # Mostramos la figura con el top 4 de vecindarios y tipos de propiedades
            fig = px.histogram(df_aux, x=df_aux['neighbourhood'], color=df_aux['property_type']
                                , title='Distribución de tipos de propiedades por vecindario', labels={'neighbourhood': 'Vecindario', 'property_type': 'Tipo de propiedad'})
            fig.update_layout(barmode='group')
            fig.update_xaxes(categoryorder='total descending', range=(-.5, 3.5))
            fig.update_yaxes(title='Número de propiedades')
            st.plotly_chart(fig, use_container_width=True)
    with tabsVecindario[1]:
        # Mostramos un gráfico con el top 4 de vecindarios con más propiedades
        fig = px.histogram(listings, x=listings['neighbourhood'], color=listings['room_type'], color_discrete_map={'Entire home/apt': '#5DADE2', 'Private room': '#239B56', 'Shared room': '#A6ACAF', "Hotel room" : "#BB8FCE"}
                            , title='Distribución de tipos de habitaciones por vecindario', labels={'neighbourhood': 'Vecindario', 'room_type': 'Tipo de habitación'})
        fig.update_layout(barmode='group', legend=dict(orientation="h", y=1.12, x=0, xanchor='left')) # Ajustamos la leyenda y el modo de ver las barras, para hacer una mejor comparación
        fig.update_xaxes(categoryorder='total descending', range=(-.5, 3.5)) # Ordenamos los vecindarios de mayor a menor y mostramos el top 4
        fig.update_yaxes(title='Número de propiedades')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        #Mapa tipos de vivienda en los cuatro barrios más populares de Sydney
        df_aux = listings.copy() # Creamos un datafarme auxiliar copiando el contenido de listings
        df_aux["other_rooms"] = df_aux["room_type"] # Creamos una nueva columna con el mismo contenido que la columna room_type
        # Iteramos sobre room_type para cambiar los valores diferentes de Entire home/apt por Other rooms
        for i in range(len(df_aux["room_type"])):
            if (df_aux.at[i, "other_rooms"] == "Shared room") or (df_aux.at[i, "other_rooms"] == "Hotel room") or (df_aux.at[i, "other_rooms"] == "Private room"):
                df_aux.at[i, "other_rooms"] = "Other rooms"

        df_aux["top4_neighbourhoods"] = df_aux[(df_aux["neighbourhood"] == "Sydney") | (df_aux["neighbourhood"] == "Waverley") | (df_aux["neighbourhood"] == "Pittwater") | (df_aux["neighbourhood"] == "Randwick")]["neighbourhood"]
        # Con los datos anteriores, creamos un mapa interactivo con las propiedades de tipo Entire home/apt y Other rooms
        fig = px.scatter_mapbox(df_aux, lat='latitude', lon='longitude', color='other_rooms', zoom=10
                                , mapbox_style='open-street-map'
                                , opacity=0.7
                                , title='Distribución de los tipos de vivienda en los cuatro barrios más populares de Sydney'
                                , labels={'room_type': 'Tipo de habitación', "other_rooms": "Clasificación", "latitude": "Latitud", "longitude": "Longitud", "price_eur": "Precio en euros", "top4_neighbourhoods": "Vecindario"}
                                , hover_data={"other_rooms": False, "top4_neighbourhoods": False, "room_type":True, "price_eur": True}
                                , color_discrete_map={'Entire home/apt': '#36454F', 'Other rooms': '#FF3131'}
                                , animation_frame="top4_neighbourhoods"
                                , height=800)
        fig.update_layout(legend=dict(orientation="h", y=1.06, x=0, xanchor='left'))
        st.plotly_chart(fig, use_container_width=True)
    with tabsVecindario[2]:
        calendar_data = pd.read_csv("https://raw.githubusercontent.com/alvaro99dd/Analisis-AirBnB-Sydney/main/Recursos/calendar.zip", low_memory=False)
        calendar_data = pd.merge(listings, calendar_data, left_on="id", right_on="listing_id", how="left")
        calendar_data = calendar_data.groupby(["neighbourhood", "date"])["available"].value_counts().unstack()
        calendar_data["available_ratio"] = np.round(calendar_data["t"] / (calendar_data["t"] + calendar_data["f"]) * 100, 2)
        calendar_data = calendar_data.reset_index()
        calendar_data = clean_outliers(calendar_data, "available_ratio")
        
        filt = st.checkbox("Mostrar todos los vecindarios", value=False)
        if filt:
            pass
        else:
            calendar_data = calendar_data[(calendar_data['neighbourhood'] == "Sydney") | (calendar_data['neighbourhood'] == "Waverley") | (calendar_data['neighbourhood'] == "Pittwater") | (calendar_data['neighbourhood'] == "Randwick")]
        #Grafica de disponibilidad de propiedades por vecindario
        fig = px.scatter(calendar_data, x= "date", y="available_ratio", title="Disponibilidad de propiedades por vecindario"
                    , color="neighbourhood"
                    , color_continuous_scale="Plasma"
                    , labels={"available_ratio": "Ratio de disponibilidad", "date": "Fecha", "neighbourhood": "Vecindario"})
        st.plotly_chart(fig, use_container_width=True)


    with tabsVecindario[3]:
        # Gráfico de barras para mostrar el precio medio por puntuación de ubicación
        df_aux = clean_outliers(df_aux, "price_eur")
        df_aux["real_reviews_location"] = df_aux[df_aux["number_of_reviews"] > 10]["review_scores_location"]
        # colors = ['#FF3131' if i >= 87 else '#36454F' for i in range(len(df_aux["real_reviews_location"]))]
        # Destacamos las propiedades que tienen una puntuación mayor o igual a 4.9
        df_aux['color_condition'] = df_aux['real_reviews_location'].apply(lambda x: 'Mayor o igual a 4.9' if x >= 4.9 else 'Menor a 4.9')
        # Mostramos el histograma con el precio medio por puntuación de ubicación
        fig = px.histogram(df_aux, x="real_reviews_location", y="price_eur"
                            , range_y=[0,500], range_x=[3.5,5], histfunc="avg"
                            , title="Precio medio por puntuación de ubicación"
                            , labels={'real_reviews_location': 'Puntuación de ubicación', 'price_eur': 'Precio en euros'}
                            , color='color_condition'
                            , color_discrete_map={'Mayor o igual a 4.9': '#FF3131', 'Menor a 4.9': '#36454F'} # Mapeo de colores
                            , hover_data={"color_condition": False})  # Ocultamos la información de color_condition en el hover
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Boxplot con el top 4 vecindarios con más puntuación de ubicación
        df_aux['color'] = df_aux['neighbourhood'].apply(lambda x: "#FF3131" if x in ["Randwick", "Sydney"] else "#36454F") # Destacamos los vecindarios Randwick y Sydney
        # Mostramos un boxplot con el top 4 vecindarios con más puntuación de ubicación, destacando Randwick y Sydney ya que pertenecen a los 4 vecindarios con más propiedades
        fig = px.box(df_aux, x="real_reviews_location", y="neighbourhood"
                            , range_x=[4,5]
                            , title="Top 4 vecindarios con más puntuación de ubicación"
                            , labels={'real_reviews_location': 'Puntuación de ubicación', 'price_eur': 'Precio en euros', "neighbourhood": "Vecindario"}
                            , orientation="h"
                            , color='color'
                            , color_discrete_map="identity")
        fig.update_yaxes(categoryorder='total descending', range=(-.5, 3.5))
        fig.update_traces(width=0.7, boxmean=True)
        st.plotly_chart(fig, use_container_width=True)
        df_aux.drop(columns=["color"], inplace=True)

def rating():
    codigo_iframe = '''<iframe title="Panel_Rating_AirBnB" width="1320" height="1240"
    src="https://app.powerbi.com/view?r=eyJrIjoiNTQzNzU5MmQtNjc0Zi00ZTA4LWEwMjktZmQ5MTYwMjA5ODRmIiwidCI6IjhhZWJkZGI2LTM0MTgtNDNhMS1hMjU1LWI5NjQxODZlY2M2NCIsImMiOjl9"
    frameborder="0" allowFullScreen="true"></iframe>'''
    html(codigo_iframe, width=1320, height=1250)

# if pestaña == "Inicio":
#     st.subheader("Investigación exhaustiva para decidir en qué propiedades y barrios es más rentable invertir")
#     cols = st.columns(2)
#     with cols[0]:
#         st.image("https://content.r9cdn.net/rimg/dimg/12/98/b1e36771-city-2258-163f4d7f814.jpg?crop=true&width=1020&height=498")
#         st.caption("Fuente: Kayak.es")
#     with cols[1]:
#         with st.expander("Resumen análisis"):
#             st.write('''
#             A lo largo de esta aplicación observaremos
#             las diferentes variables que afectan no sólo al precio
#             medio de la vivienda sino también a sus puntuaciones en la aplicación
#             AirBnB, sirviendo como guía o consejos para nuestros clientes
#             a la hora de invertir en la zona de Sydney.
#             ''')
#             col1, col2, col3 = st.columns([1,2,1])
#             with col2:
#                 st.image("https://www.eleconomista.es/finanzas-personales/wp-content/uploads/2023/12/Untitled-1-1.png", width=300)
#                 st.caption("Fuente: El Economista")
#             # st.markdown("##### Datos analizados")
#             # st.markdown("##### Precio medio")
#             # st.markdown("##### Vecindarios")
#             # st.markdown("##### Rating")

# elif pestaña == "Datos usados":
#     tabsInicio = st.tabs(["Datos Cargados"])
#     with tabsInicio[0]:
#         filtrotabla = st.checkbox("Mostrar datos analizados", value=False)
#         if filtrotabla:
#             df = pd.DataFrame(data=listings, columns=['neighbourhood', 'price_eur', 'room_type', 'property_type',
#             'number_of_reviews', 'review_scores_rating', 'amenities', 'accommodates', 'review_scores_location', 'bedrooms'])
#             st.subheader("Datos Analizados")
#             st.dataframe(df)
#         else:
            
#             st.subheader("Datos Preprocesados")
#             st.dataframe(listings)
    # with tabsInicio[1]:
    #     # @st.cache_data
    #     def create_map(locations):
    #         map1 = folium.Map(location=[-33.86785, 151.20732], zoom_start=11.5)
    #         FastMarkerCluster(data=locations).add_to(map1)
    #         return map1
    #     # st.cache_data(create_map)
        
    #     # Creamos una lista con las latitudes y longitudes de las propiedades en Sydney
    #     lats2018 = listings['latitude'].tolist()
    #     lons2018 = listings['longitude'].tolist()
    #     locations = list(zip(lats2018, lons2018))

    #     # Creamos el mapa utilizando la función cacheada
    #     map1 = create_map(locations)

    #     # Mostramos el mapa en Streamlit
    #     st_folium(map1, width='100%')
        
    #     # st.image("https://www.sydney.com/sites/sydney/files/styles/full_height_hero/public/2019-09/hero_sydney_skyline_2019_credit_destination_nsw.jpg?itok=3Z9J9Q9Y", width=1320)

# elif pestaña == "Importancia del Precio":
#     tabsPrecio = st.tabs(["Según propiedad", "Según valoraciones", "Según comodidades", "Según barrio"])
#     with tabsPrecio[0]:
#         cols = st.columns(2)
#         with cols[0]:
#             st.markdown("#####")
#             # Grafica precio medio por tipo de propiedad
#             mean_price = listings.groupby('property_type')['price_eur'].mean().round(2).sort_values(ascending=False).head(10)
#             colors = ['#FF3131' if i < 1 else '#36454F' for i in range(len(mean_price))]

#             fig = px.bar(mean_price, x=mean_price.index, y=mean_price.values, labels={'y':'Precio Medio', 'property_type':'Tipo de Propiedad'})
#             fig.update_layout(
#                 xaxis_title='Tipo de Propiedad',
#                 yaxis_title='Precio Medio',
#                 title='Precio Medio por Tipo de Propiedad'
#             )
#             fig.update_traces(marker_color=colors)
#             st.plotly_chart(fig, use_container_width=True)
#         with cols[1]:
#             # Grafica de barras para mostrar el precio medio por número de habitaciones con el top 5 tipos de propiedad
#             df_aux = listings.copy()
#             grouped_df = df_aux.groupby(['bedrooms', 'property_type']).agg(
#             price_eur_mean=('price_eur', 'mean')
#             ).reset_index()

#             top_5_property_types = df_aux['property_type'].value_counts().nlargest().index

#             filtered_grouped_df = grouped_df[grouped_df['property_type'].isin(top_5_property_types)]

#             final_filtered_df = filtered_grouped_df[(filtered_grouped_df['bedrooms'] >= 1)&(filtered_grouped_df['bedrooms'] <= 10)]
#             fig = px.bar(final_filtered_df,
#             x="bedrooms",
#             y="price_eur_mean",
#             color="property_type",
#             barmode="group",
#             title="Precio Medio(€) por Número de Habitaciones con el Top 5 Tipos de Propiedad",
#             labels={"bedrooms": "Número de Habitaciones", "price_eur_mean": "Precio Medio (EUR)", "property_type": "Tipo de propiedad"},
#             category_orders={"bedrooms": sorted(final_filtered_df['bedrooms'].unique())})  # Ordenar las categorías de 'bedrooms'

#             fig.update_layout(
#             plot_bgcolor="white",
#             yaxis=dict(title='Precio Medio (EUR)', gridcolor='lightgrey'),
#             xaxis=dict(title='Número de Habitaciones'),
#             legend=dict(title='Tipo de Propiedad')
#             )
#             st.plotly_chart(fig, use_container_width=True)
#     with tabsPrecio[1]:
#         # Mapa de propiedades por Precio que tengan mas de 10 Reseñas
#         df_filtered = listings.copy()
#         df_filtered['review_scores_rating'] = listings[listings["number_of_reviews"] > 10]["review_scores_rating"]

#         df_filtered = df_filtered.dropna(subset=["review_scores_rating"])
#         df_filtered = df_filtered.sort_values(by="review_scores_rating", ascending=True)
#         df_filtered['adjusted_size'] = df_filtered['price_eur'] + 35

#         fig = px.scatter_mapbox(df_filtered, lat="latitude", lon="longitude",
#         color="price_eur", size="adjusted_size",
#         size_max=55, 
#         animation_frame="review_scores_rating",
#         zoom=9, mapbox_style="open-street-map",
#         color_continuous_scale="viridis",
#         title="Propiedades por Precio que tengan mas de 10 Reseñas", range_color=[0, 500],
#         labels={"price_eur": "Precio (EUR)", "review_scores_rating": "Valoración Total"},
#         height=800)
#         st.plotly_chart(fig, use_container_width=True)

#         # Grafica districución todas las reviews, valoración total y precio
#         fig = px.scatter(listings, x='review_scores_rating', y='number_of_reviews', color="price_eur",
#         size="number_of_reviews", range_color=[0, 500], range_x=[4, 5],
#         title="Distribución todas las Valoraciones, Valoración Total y Precio",
#         color_continuous_scale="Portland",
#         labels={"review_scores_rating": "Valoración Total", "number_of_reviews": "Total Reviews", "price_eur": "Precio(€)"},)
#         st.plotly_chart(fig, use_container_width=True)
#     with tabsPrecio[2]:
#         #Grafica total de comodidades por numero de propiedades
#         df_aux = listings.copy()
#         df_aux['total_amenities'] = df_aux['amenities'].apply(lambda x: len(x.split(',')) if x != '[]' else 0)

#         mean_price_amenities=df_aux.groupby('total_amenities')['price_eur'].mean().round(2).reset_index(name='mean_price')
#         count_properties_amenities = df_aux.groupby('total_amenities')['price_eur'].count().reset_index(name='count_properties')

#         result_df = pd.merge(mean_price_amenities, count_properties_amenities, on='total_amenities')

#         fig = px.scatter(result_df, x='total_amenities', y='count_properties', color="mean_price", size="count_properties", title='Numero de propiedades por total de comodidades y precio medio', range_color=[0, 500],
#                         labels={'mean_price': 'Precio Medio(€)', 'count_properties': 'Numero de propiedades', 'total_amenities': 'Total de comodidades'}, color_continuous_scale='Plasma')
#         fig.update_layout(
#             xaxis_title='Total de comodidades',
#             yaxis_title='Numero de propiedades',
#             title='Total de comodidades por numero de propiedades',
#         )
#         st.plotly_chart(fig, use_container_width=True)
#     with tabsPrecio[3]:
#         listings_clean_df = listings.copy()
#         listings_clean_df.dropna(subset=['neighbourhood', 'price_eur'], inplace=True)
#         price_mean_by_neighbourhood = listings_clean_df.groupby('neighbourhood', as_index=False)['price_eur'].mean().round(2)
#         price_mean_by_neighbourhood.rename(columns={'price_eur': 'price_mean_eur'}, inplace=True)

#         fig = px.treemap(price_mean_by_neighbourhood, 
#         path=['neighbourhood'], 
#         values='price_mean_eur',
#         color='price_mean_eur', 
#         color_continuous_scale='RdYlGn',
#         title='Distribución de Precio Medio por Barrio',
#         labels={'price_mean_eur': 'Precio Medio (€)'})
#         fig.update_traces(hovertemplate='Barrio: %{label}<br>Precio Medio (€): %{value}')
#         st.plotly_chart(fig, use_container_width=True)
# elif pestaña == "Importancia del Vecindario":
#     tabsVecindario = st.tabs(["Propiedades según vecindario", "Distribución habitaciones", "Disponibilidad", "Según puntuación de ubicación"])
#     with tabsVecindario[0]:
#         cols = st.columns(2)
#         with cols[0]:
#             st.markdown("<br/>", unsafe_allow_html=True)
#             # Gráfico de barras para mostrar el top 10 de vecindarios con más propiedades
#             feq=listings['neighbourhood'].value_counts(ascending=False).head(10) # Calculamos el top 10 de vecindarios con más propiedades
#             colors = ['#FF3131' if i < 4 else '#36454F' for i in range(len(feq))] # Destacamos el top 4
#             # Creamos y mostramos el gráfico
#             fig = feq.plot.barh(figsize=(10, 5), color=colors, width=1, subplots=True)
#             plt.title("Top 10 vecindarios con más propiedades", fontsize=20)
#             plt.xlabel('Número de propiedades', fontsize=12)
#             plt.ylabel('', fontsize=12)

#             st.pyplot(plt)
#         with cols[1]:
#             #Tipo de propiedades según el top 4 de vecindarios
#             # Creamos un dataframe auxiliar con el top 4 visto anteriormente
#             df_aux = listings[(listings['neighbourhood'] == "Sydney") | (listings['neighbourhood'] == "Waverley") | (listings['neighbourhood'] == "Pittwater") | (listings['neighbourhood'] == "Randwick")]
#             # Mostramos la figura con el top 4 de vecindarios y tipos de propiedades
#             fig = px.histogram(df_aux, x=df_aux['neighbourhood'], color=df_aux['property_type']
#                                 , title='Distribución de tipos de propiedades por vecindario', labels={'neighbourhood': 'Vecindario', 'property_type': 'Tipo de propiedad'})
#             fig.update_layout(barmode='group')
#             fig.update_xaxes(categoryorder='total descending', range=(-.5, 3.5))
#             fig.update_yaxes(title='Número de propiedades')
#             st.plotly_chart(fig, use_container_width=True)
#     with tabsVecindario[1]:
#         # Mostramos un gráfico con el top 4 de vecindarios con más propiedades
#         fig = px.histogram(listings, x=listings['neighbourhood'], color=listings['room_type'], color_discrete_map={'Entire home/apt': '#5DADE2', 'Private room': '#239B56', 'Shared room': '#A6ACAF', "Hotel room" : "#BB8FCE"}
#                             , title='Distribución de tipos de habitaciones por vecindario', labels={'neighbourhood': 'Vecindario', 'room_type': 'Tipo de habitación'})
#         fig.update_layout(barmode='group', legend=dict(orientation="h", y=1.12, x=0, xanchor='left')) # Ajustamos la leyenda y el modo de ver las barras, para hacer una mejor comparación
#         fig.update_xaxes(categoryorder='total descending', range=(-.5, 3.5)) # Ordenamos los vecindarios de mayor a menor y mostramos el top 4
#         fig.update_yaxes(title='Número de propiedades')
#         st.plotly_chart(fig, use_container_width=True)

#         st.markdown("---")

#         #Mapa tipos de vivienda en los cuatro barrios más populares de Sydney
#         df_aux = listings.copy() # Creamos un datafarme auxiliar copiando el contenido de listings
#         df_aux["other_rooms"] = df_aux["room_type"] # Creamos una nueva columna con el mismo contenido que la columna room_type
#         # Iteramos sobre room_type para cambiar los valores diferentes de Entire home/apt por Other rooms
#         for i in range(len(df_aux["room_type"])):
#             if (df_aux.at[i, "other_rooms"] == "Shared room") or (df_aux.at[i, "other_rooms"] == "Hotel room") or (df_aux.at[i, "other_rooms"] == "Private room"):
#                 df_aux.at[i, "other_rooms"] = "Other rooms"

#         df_aux["top4_neighbourhoods"] = df_aux[(df_aux["neighbourhood"] == "Sydney") | (df_aux["neighbourhood"] == "Waverley") | (df_aux["neighbourhood"] == "Pittwater") | (df_aux["neighbourhood"] == "Randwick")]["neighbourhood"]
#         # Con los datos anteriores, creamos un mapa interactivo con las propiedades de tipo Entire home/apt y Other rooms
#         fig = px.scatter_mapbox(df_aux, lat='latitude', lon='longitude', color='other_rooms', zoom=10
#                                 , mapbox_style='open-street-map'
#                                 , opacity=0.7
#                                 , title='Distribución de los tipos de vivienda en los cuatro barrios más populares de Sydney'
#                                 , labels={'room_type': 'Tipo de habitación', "other_rooms": "Clasificación", "latitude": "Latitud", "longitude": "Longitud", "price_eur": "Precio en euros", "top4_neighbourhoods": "Vecindario"}
#                                 , hover_data={"other_rooms": False, "top4_neighbourhoods": False, "room_type":True, "price_eur": True}
#                                 , color_discrete_map={'Entire home/apt': '#36454F', 'Other rooms': '#FF3131'}
#                                 , animation_frame="top4_neighbourhoods"
#                                 , height=800)
#         fig.update_layout(legend=dict(orientation="h", y=1.06, x=0, xanchor='left'))
#         st.plotly_chart(fig, use_container_width=True)
#     with tabsVecindario[2]:
#         calendar_data = pd.read_csv("https://raw.githubusercontent.com/alvaro99dd/Analisis-AirBnB-Sydney/main/Recursos/calendar.zip", low_memory=False)
#         calendar_data = pd.merge(listings, calendar_data, left_on="id", right_on="listing_id", how="left")
#         calendar_data = calendar_data.groupby(["neighbourhood", "date"])["available"].value_counts().unstack()
#         calendar_data["available_ratio"] = np.round(calendar_data["t"] / (calendar_data["t"] + calendar_data["f"]) * 100, 2)
#         calendar_data = calendar_data.reset_index()
#         calendar_data = clean_outliers(calendar_data, "available_ratio")
        
#         filt = st.checkbox("Mostrar todos los vecindarios", value=False)
#         if filt:
#             pass
#         else:
#             calendar_data = calendar_data[(calendar_data['neighbourhood'] == "Sydney") | (calendar_data['neighbourhood'] == "Waverley") | (calendar_data['neighbourhood'] == "Pittwater") | (calendar_data['neighbourhood'] == "Randwick")]
#         #Grafica de disponibilidad de propiedades por vecindario
#         fig = px.scatter(calendar_data, x= "date", y="available_ratio", title="Disponibilidad de propiedades por vecindario"
#                     , color="neighbourhood"
#                     , color_continuous_scale="Plasma"
#                     , labels={"available_ratio": "Ratio de disponibilidad", "date": "Fecha", "neighbourhood": "Vecindario"})
#         st.plotly_chart(fig, use_container_width=True)


#     with tabsVecindario[3]:
#         # Gráfico de barras para mostrar el precio medio por puntuación de ubicación
#         df_aux = clean_outliers(df_aux, "price_eur")
#         df_aux["real_reviews_location"] = df_aux[df_aux["number_of_reviews"] > 10]["review_scores_location"]
#         # colors = ['#FF3131' if i >= 87 else '#36454F' for i in range(len(df_aux["real_reviews_location"]))]
#         # Destacamos las propiedades que tienen una puntuación mayor o igual a 4.9
#         df_aux['color_condition'] = df_aux['real_reviews_location'].apply(lambda x: 'Mayor o igual a 4.9' if x >= 4.9 else 'Menor a 4.9')
#         # Mostramos el histograma con el precio medio por puntuación de ubicación
#         fig = px.histogram(df_aux, x="real_reviews_location", y="price_eur"
#                             , range_y=[0,500], range_x=[3.5,5], histfunc="avg"
#                             , title="Precio medio por puntuación de ubicación"
#                             , labels={'real_reviews_location': 'Puntuación de ubicación', 'price_eur': 'Precio en euros'}
#                             , color='color_condition'
#                             , color_discrete_map={'Mayor o igual a 4.9': '#FF3131', 'Menor a 4.9': '#36454F'} # Mapeo de colores
#                             , hover_data={"color_condition": False})  # Ocultamos la información de color_condition en el hover
#         fig.update_layout(showlegend=False)
#         st.plotly_chart(fig, use_container_width=True)

#         st.markdown("---")

#         # Boxplot con el top 4 vecindarios con más puntuación de ubicación
#         df_aux['color'] = df_aux['neighbourhood'].apply(lambda x: "#FF3131" if x in ["Randwick", "Sydney"] else "#36454F") # Destacamos los vecindarios Randwick y Sydney
#         # Mostramos un boxplot con el top 4 vecindarios con más puntuación de ubicación, destacando Randwick y Sydney ya que pertenecen a los 4 vecindarios con más propiedades
#         fig = px.box(df_aux, x="real_reviews_location", y="neighbourhood"
#                             , range_x=[4,5]
#                             , title="Top 4 vecindarios con más puntuación de ubicación"
#                             , labels={'real_reviews_location': 'Puntuación de ubicación', 'price_eur': 'Precio en euros', "neighbourhood": "Vecindario"}
#                             , orientation="h"
#                             , color='color'
#                             , color_discrete_map="identity")
#         fig.update_yaxes(categoryorder='total descending', range=(-.5, 3.5))
#         fig.update_traces(width=0.7, boxmean=True)
#         st.plotly_chart(fig, use_container_width=True)
#         df_aux.drop(columns=["color"], inplace=True)    
# elif pestaña == "Importancia del rating":
#     codigo_iframe = '''<iframe title="Panel_Rating_AirBnB" width="1320" height="1240"
#     src="https://app.powerbi.com/view?r=eyJrIjoiNTQzNzU5MmQtNjc0Zi00ZTA4LWEwMjktZmQ5MTYwMjA5ODRmIiwidCI6IjhhZWJkZGI2LTM0MTgtNDNhMS1hMjU1LWI5NjQxODZlY2M2NCIsImMiOjl9"
#     frameborder="0" allowFullScreen="true"></iframe>'''
#     html(codigo_iframe, width=1320, height=1250)