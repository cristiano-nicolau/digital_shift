import os
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from text_model import ProductAnalyzer, ProductVisualizer, TimeSeriesAnalyzer, ProductPricePrediction  
from datetime import timedelta

def main():

    def gerar_wordcloud(texto):
        wordcloud = WordCloud(width=600, height=300, background_color='white', max_words=500).generate(texto)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')  
        st.pyplot(fig) 

    st.set_page_config(layout="wide") 

    tab_selection = st.sidebar.radio(
        "Select a Tab",
        ["Price Predictor","Web Site Graph","Web Site WordCloud","Product and Price Analysis", "Time Series Analysis"],
        index=0
    )

    if tab_selection == "Price Predictor":

        df = pd.read_csv('ml_dataset.csv')

        st.subheader("Select a Product to Predict the Price")
        col1, col2 = st.columns(2)
        with col1:
            sites = df['site'].dropna().unique()
            site = st.selectbox("Select the Site", sites)
            categories = df['categoria'].dropna().unique()
            category = st.selectbox("Select the Category", categories)
            colors = df['color'].dropna().unique()
            color = st.selectbox("Select the Color", colors)
            ram_sizes = df['RAM_size'].dropna().unique()
            ram_size = st.selectbox("Select the RAM Size", ram_sizes)

        with col2:
            brands = df[df['categoria'] == category]['brand'].dropna().unique()
            brand = st.selectbox("Select the Brand", brands)
            models = df[df['brand'] == brand]['model'].dropna().unique()
            model = st.selectbox("Select the Model", models)
            hdd_sizes = df['HDD_size'].dropna().unique()
            hdd_size = st.selectbox("Select the HDD Size", hdd_sizes)
            left, middle, right = st.columns(3)
            button = middle.button("Predict Price",use_container_width=True)

        predictor = ProductPricePrediction(df)
        predictor.train_model()
        if button:
            result = {
                'site': site,
                'categoria': category,
                'brand': brand,
                'brand_model': model,
                'color': color,
                'HDD_size': hdd_size,
                'RAM_size': ram_size
            }

            result['promocao'] = 0
            price_without_promotion = predictor.predict_price(result)

            result['promocao'] = 1
            price_with_promotion = predictor.predict_price(result)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Price Without Promotion")
                st.error(f"€ {price_without_promotion:.2f}")
            with col2:
                st.subheader("Price With Promotion")
                st.success(f"€ {price_with_promotion:.2f}")


    elif tab_selection == "Web Site Graph":
        
        df = pd.read_csv('all_prices_extracted_cleaned_categorized_analyzed.csv')

        cores_disponiveis = px.colors.qualitative.Bold
        categorias_unicas = df['categoria'].dropna().unique()
        categoria_cores = {
            categoria: cores_disponiveis[i % len(cores_disponiveis)]
            for i, categoria in enumerate(categorias_unicas)
        }

        def criar_grafo_interativo(df, site, categorias_selecionadas):
            edges = []
            df_filtrado = df[(df['site'] == site) & (df['categoria'].isin(categorias_selecionadas))]

            for _, row in df_filtrado.iterrows():
                brand = row['brand']
                model = row['model']
                submodel = row['brand_model']

                if pd.notna(brand) and pd.notna(model):
                    edges.append((brand, model))

                if pd.notna(model) and pd.notna(submodel):
                    edges.append((model, submodel))

            G = nx.DiGraph()
            G.add_edges_from(edges)

            pos = nx.spring_layout(G, seed=42)

            x_nodes = [pos[node][0] for node in G.nodes()]
            y_nodes = [pos[node][1] for node in G.nodes()]

            edge_x = []
            edge_y = []
            edge_hover_text = []  
            edge_text = []  

            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

                edge_text.append(f"{edge[0]} → {edge[1]}")
                edge_hover_text.append(f"{edge[0]} → {edge[1]}")

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='gray'),
                hoverinfo='text',
                text=edge_hover_text,
                mode='lines',
                textposition='middle center',  
                textfont=dict(size=10, color='white')  
            )

            node_trace = go.Scatter(
                x=x_nodes, y=y_nodes,
                mode='markers+text',
                hoverinfo='text',
                marker=dict(
                    showscale=False,
                    colorscale='Viridis',
                    size=20,
                    line_width=2
                ),
                text=[],  
                textposition='top center', 
                textfont=dict(size=12, color='white') 
            )

            node_text = []  
            node_labels = []  
            node_colors = []  

            for node in G.nodes():
                if node in df_filtrado['brand'].unique():
                    count = len(df_filtrado[df_filtrado['brand'] == node])
                    node_text.append(f"Marca: {node}<br>Produtos: {count}")
                    node_labels.append(node) 
                    node_colors.append('skyblue')  
                elif node in df_filtrado['model'].unique():
                    categoria = df_filtrado[df_filtrado['model'] == node]['categoria'].values[0]
                    count = len(df_filtrado[df_filtrado['model'] == node])
                    node_text.append(f"Modelo: {node}<br>Produtos: {count}<br>Categoria: {categoria}")
                    node_labels.append(node) 
                    node_colors.append(categoria_cores[categoria])  
                elif node in df_filtrado['brand_model'].unique():
                    categoria = df_filtrado[df_filtrado['brand_model'] == node]['categoria'].values[0]
                    count = len(df_filtrado[df_filtrado['brand_model'] == node])
                    node_text.append(f"Submodelo: {node}<br>Produtos: {count}<br>Categoria: {categoria}")
                    node_labels.append(node)  
                    node_colors.append(categoria_cores[categoria])  
                else:
                    node_text.append(node)
                    node_labels.append(node)
                    node_colors.append('lightgray')

            node_trace.marker.color = node_colors
            node_trace.text = node_labels 
            node_trace.hovertext = node_text  
            node_trace.hoverinfo = 'text' 

            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f"Grafo Interativo para o Site: {site}",
                    titlefont=dict(size=18),
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=40),
                    annotations=[
                        dict(
                            text="Passe o mouse pelos nós e arestas para mais informações.",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )
                    ],
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False)
                )
            )

            for categoria in categorias_selecionadas:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=15, color=categoria_cores[categoria]),
                    legendgroup=categoria,
                    showlegend=True,
                    name=categoria,

                ))

            return fig


        st.title("Visualization of the Web Site Graph")

        sites_disponiveis = df['site'].dropna().unique()
        site_selecionado = st.selectbox("Select the site:", sites_disponiveis)

        categorias_selecionadas = st.multiselect("Select Categories", categorias_unicas, default=categorias_unicas)

        fig_site = criar_grafo_interativo(df, site_selecionado, categorias_selecionadas)
        st.plotly_chart(fig_site)


    elif tab_selection == "Web Site WordCloud":
        df = pd.read_csv('all_prices_extracted_cleaned_categorized_analyzed.csv')

        st.title("Word Cloud by Site")

        sites_disponiveis = df['site'].unique()
        site_selecionado = st.selectbox("Select your site", sites_disponiveis)

        categorias = df[df['site'] == site_selecionado]['categoria'].dropna().astype(str).unique()
        marcas = df[df['site'] == site_selecionado]['brand'].dropna().astype(str).unique()
        modelos = df[df['site'] == site_selecionado]['model'].dropna().astype(str).unique()
        submodelos = df[df['site'] == site_selecionado]['brand_model'].dropna().astype(str).unique()
        cores = df[df['site'] == site_selecionado]['color'].dropna().astype(str).unique()

        texto = ' '.join(set(categorias) | set(marcas) | set(modelos) | set(submodelos) | set(cores))

        st.subheader(f'Word Cloud - Site: {site_selecionado}')
        gerar_wordcloud(texto)

    elif tab_selection == "Product and Price Analysis":
        st.title("Product and Price Analysis")

        uploaded_file = ("all_prices_extracted_cleaned_categorized.csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            analyzer = ProductAnalyzer()
            df_analyzed, insights = analyzer.analyze_dataset(df)

            st.title("Visualizations")

            ### ----------------------------------------------- ####  
            st.subheader("Product Distribution by Category")
            sites_disponiveis = df_analyzed['site'].dropna().unique()
            site_selecionado = st.multiselect(
                "Select your Sites:",
                options=sites_disponiveis,
                default=sites_disponiveis
            )
            df_filtrado = df_analyzed[df_analyzed['site'].isin(site_selecionado)]
            visualizer_product_destribution = ProductVisualizer(df_filtrado, insights)
            fig = visualizer_product_destribution.plot_category_distribution()  
            st.plotly_chart(fig)

            #### ----------------------------------------------- ####  
            st.subheader("Market Share by Brand and Category")
            sites_disponiveis_to_marketshare = df_analyzed['site'].dropna().unique()
            site_selecionado_to_marketshare = st.multiselect(
                "Select your Sites:",
                options=sites_disponiveis_to_marketshare,
                default=sites_disponiveis_to_marketshare,
                key="marketshare_sites"
            )
            df_filtrado = df_analyzed[df_analyzed['site'].isin(site_selecionado_to_marketshare)]
            categorias_disponiveis = df_filtrado['categoria'].dropna().unique()
            categoria_selecionada = st.multiselect(
                "Select your Categories:",
                options=categorias_disponiveis,
                default=categorias_disponiveis,
                key="marketshare_categories"
            )
            df_filtrado = df_filtrado[df_filtrado['categoria'].isin(categoria_selecionada)]
            visualizer_market_share = ProductVisualizer(df_filtrado, insights)          
            fig = visualizer_market_share.plot_brand_market_share()  
            st.plotly_chart(fig)

            #### ----------------------------------------------- ####  

            st.subheader("Mean Price Comparison by Brand and Category")
            site_selecionado_to_price_comparison = df_analyzed['site'].dropna().unique()
            site_selecionado_to_price = st.multiselect(
                "Select your Sites:",
                options=site_selecionado_to_price_comparison,
                default=site_selecionado_to_price_comparison,
                key="price_comparison_sites"
            )
            df_filtrado = df_analyzed[df_analyzed['site'].isin(site_selecionado_to_price)]
            categorias_disponiveis = df_filtrado['categoria'].dropna().unique()
            categoria_selecionada = st.multiselect(
                "Select your Categories:",
                options=categorias_disponiveis,
                default=categorias_disponiveis,
                key="price_comparison_categories"
            )
            df_filtrado = df_filtrado[df_filtrado['categoria'].isin(categoria_selecionada)]
            visualizer = ProductVisualizer(df_filtrado, insights)
            fig = visualizer.plot_price_comparison()  
            st.plotly_chart(fig)

            #### ----------------------------------------------- ####  

            st.subheader("Brand Presence by Site")
            site_to_brand_presence = df_analyzed['site'].dropna().unique()
            site_selecionado_to_brand_presence = st.multiselect(
                "Select your Sites:",
                options=site_to_brand_presence,
                default=site_to_brand_presence,
                key="brand_presence_sites"
            )
            df_filtrado = df_analyzed[df_analyzed['site'].isin(site_selecionado_to_brand_presence)]
            categorias_disponiveis_brand_presence = df_filtrado['categoria'].dropna().unique()
            categoria_selecionada_brand_presence = st.multiselect(
                "Select your Categories:",
                options=categorias_disponiveis_brand_presence,
                default=categorias_disponiveis_brand_presence,
                key="brand_presence_categories"
            )
            df_filtrado2 = df_filtrado[df_filtrado['categoria'].isin(categoria_selecionada_brand_presence)]
            visualizer = ProductVisualizer(df_filtrado2, insights)
            fig = visualizer.plot_site_brand_presence()  
            st.plotly_chart(fig)

            #### ----------------------------------------------- ####
            st.subheader("Distribution of Product Models")

            # Seleção de sites
            sites_disponiveis = df_analyzed['site'].dropna().unique()
            sites_selecionados = st.multiselect(
                "Select Sites:",
                options=sites_disponiveis,
                default=sites_disponiveis,
                key="sites_filter"
            )

            categorias_disponiveis = df_analyzed['categoria'].dropna().unique()
            categorias_selecionadas = st.multiselect(
                "Select Categories:",
                options=categorias_disponiveis,
                default=categorias_disponiveis,
                key="categories_filter"
            )

            df_filtrado = df_analyzed[
                df_analyzed['site'].isin(sites_selecionados) & 
                df_analyzed['categoria'].isin(categorias_selecionadas)
            ]

            visualizer = ProductVisualizer(df_filtrado, insights)
            fig = visualizer.plot_model_series_distribution_iphone8()  
            st.plotly_chart(fig)


            #### ----------------------------------------------- ####

            st.subheader("Temporal Price Comparison by Category")
            categorias_disponiveis = df_analyzed['categoria'].dropna().unique()
            categorias_selecionadas = st.multiselect(
                "Select Categories:",
                options=categorias_disponiveis,
                default=categorias_disponiveis,
                key="temporal_price_categories"
            )
            df_filtrado = df_analyzed[
                df_analyzed['categoria'].isin(categorias_selecionadas)
            ]
            visualizer = ProductVisualizer(df_filtrado, insights)
            fig = visualizer.compare_temporal_by_category()
            st.plotly_chart(fig)


            #### ----------------------------------------------- ####

            visualizer = ProductVisualizer(df_analyzed, insights)
            st.subheader("Discount Analysis by Brand")
            fig = visualizer.plot_discount_analysis()  
            st.plotly_chart(fig)


            #### ----------------------------------------------- ####

            st.subheader("Discount Distribution Analysis")
            categorias_disponiveis = df_analyzed['categoria'].dropna().unique()
            categorias_selecionadas = st.multiselect(
                "Select Categories:",
                options=categorias_disponiveis,
                default=categorias_disponiveis,
                key="discount_categories"
            )
            df_filtrado_descontos = df_analyzed[
                df_analyzed['categoria'].isin(categorias_selecionadas)
            ]
            visualizer = ProductVisualizer(df_filtrado_descontos, insights)
            fig = visualizer.analyze_discount_distribution()
            st.plotly_chart(fig)

            #### ----------------------------------------------- ####
            st.subheader("Price and Discount Correlation")
            sites_disponiveis = df_analyzed['site'].dropna().unique()
            sites_selecionados = st.multiselect(
                "Select Sites:",
                options=sites_disponiveis,
                default=sites_disponiveis,
                key="price_discount_sites"
            )

            # Filtrar os dados pelo site selecionado
            df_filtrado = df_analyzed[
                df_analyzed['site'].isin(sites_selecionados) & 
                df_analyzed['discount_percent'] > 0
            ]
            visualizer = ProductVisualizer(df_filtrado, insights)
            fig2 = visualizer.price_discount_correlation()
            st.plotly_chart(fig2)

            #### ----------------------------------------------- ####

            st.subheader("Most Discounted Products")
            visualizer = ProductVisualizer(df_analyzed, insights)
            num_products_options = [1, 5, 10, 20]  # Opções para o usuário escolher o número de promoções
            num_products = st.selectbox(
                "Select number of most discounted products to view:",
                options=num_products_options,
                index=2,  # Valor padrão (10)
                key="num_most_discounted"
            )

            # Criando seleção de sites
            sites_disponiveis = df_analyzed['site'].dropna().unique()
            selected_sites = st.multiselect(
                "Select the sites to focus on:",
                options=sites_disponiveis,
                default=sites_disponiveis,
                key="selected_sites"
            )

            # Gerando o gráfico com as seleções feitas pelo usuário
            fig1 = visualizer.products_most_discounted(num_products, selected_sites)
            st.plotly_chart(fig1)




    elif tab_selection == "Time Series Analysis":
        df = pd.read_csv('dataset_produts_to_ml.csv')
        analyzer = TimeSeriesAnalyzer(df)

        category_selection = st.selectbox("Select a Category", df['categoria'].unique())
        filtered_df = df[df['categoria'] == category_selection]

        product_selection = st.selectbox(
            "Select a Product",
            filtered_df.apply(lambda x: f"{x['brand']} {x['brand_model']}", axis=1).unique()
        )

        brand, brand_model = product_selection.split(' ', 1)  

        selected_product_df = filtered_df[
            (filtered_df['brand'] == brand) & 
            (filtered_df['brand_model'] == brand_model)
        ]
        selected_product = selected_product_df.iloc[0]
        result = analyzer.analyze_product(
            brand=selected_product['brand'],
            model=selected_product['model'],
            color=selected_product['color'],
            category=category_selection,
            brand_model=selected_product['brand_model'],
        )

        if result:
            time_series = result['time_series']
            forecasts = result['forecasts']

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=time_series.index,
                y=time_series['min_price'],
                mode='lines',
                name='Historical Data'
            ))

            forecast_dates = pd.date_range(
                start=time_series.index[-1] + timedelta(days=30), 
                periods=len(forecasts['forecast']),
                freq='MS'
            )
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecasts['forecast'],
                mode='lines',
                name='SARIMA Forecast',
                line=dict(dash='dash', color='orange')
            ))

            fig.add_trace(go.Scatter(
                x=list(forecast_dates) + list(forecast_dates[::-1]),
                y=list(forecasts['conf_int'][:, 0]) + list(forecasts['conf_int'][:, 1][::-1]),
                fill='toself',
                fillcolor='rgba(255, 165, 0, 0.2)',
                line=dict(color='rgba(255, 165, 0, 0)'),
                hoverinfo="skip",
                name='Confidence Interval'
            ))

            fig.update_layout(
                title=f"Forecast for {product_selection} - {selected_product['color']} ({category_selection})",
                xaxis_title="Date",
                yaxis_title="Price",
                legend_title="Legend",
                template="plotly_white"
            )

            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
