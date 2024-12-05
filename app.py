import os
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from text_model import ProductAnalyzer, ProductVisualizer  

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
        ["Web Site Graph","Web Site WordCloud","Product and Price Analysis"],
        index=0
    )


    if tab_selection == "Web Site Graph":
        
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

            visualizer = ProductVisualizer(df_analyzed, insights)
    
            f = pd.read_csv('all_prices_extracted_cleaned_categorized_analyzed.csv')
            site_brand_counts = f.groupby('site')['brand'].nunique().sort_values(ascending=False)
            st.write("Sites with the Most Unique Brands:")
            st.dataframe(site_brand_counts)
            
            brand_site_counts = f.groupby('brand')['site'].nunique().sort_values(ascending=False)
            st.write("Brands Available on the Most Sites:")
            st.dataframe(brand_site_counts)
            
            st.write("Brands Associated with Each Site")
            site_brands = f.groupby('site')['brand'].apply(lambda x: list(set(x))).reset_index(name='brands')
            st.write(site_brands)
            
            st.write("Models Associated with Each Brand")
            brand_models = f.groupby('brand')['brand_model'].apply(lambda x: list(set(x))).reset_index(name='models')
            st.write(brand_models)
            
            st.write("Categories Associated with Each Model")
            model_categories = f.groupby('model')['categoria'].apply(lambda x: list(set(x))).reset_index(name='categories')
            st.write(model_categories)

            st.title("Visualizations")

            st.subheader("Product Distribution by Category")
            fig = visualizer.plot_category_distribution()  
            st.plotly_chart(fig)
            st.write("This chart shows the distribution of products across various categories, helping to identify the most popular product types.")

            st.subheader("Market Share by Brand and Category")
            fig = visualizer.plot_brand_market_share()  
            st.plotly_chart(fig)
            st.write("The market share chart provides insights into which brands dominate specific categories.")

            st.subheader("Price Comparison by Brand and Category")
            fig = visualizer.plot_price_comparison()  
            st.plotly_chart(fig)
            st.write("Here, we compare the price ranges of different brands within each category to identify pricing trends.")

            st.subheader("Discount Analysis by Brand")
            visualizer.plot_discount_analysis()  
            st.pyplot(plt)
            st.write("This analysis highlights the average discount rates for each brand, revealing where customers can find the best deals.")

            st.subheader("Brand Presence by Site")
            visualizer.plot_site_brand_presence()  
            st.pyplot(plt)
            st.write("This visualization showcases the presence of various brands across different sites, reflecting the diversity of offerings.")

            st.subheader("Distribution of Product Models")
            fig = visualizer.plot_model_series_distribution_iphone8()  
            st.plotly_chart(fig)
            st.write("This chart focuses on the distribution of specific product models, such as iPhone 8, across the dataset.")

            st.subheader("Temporal Price Analysis for iPhone 8")
            fig = visualizer.analyze_temporal_for_product("apple", "iphone 8", 64.0, "prateado", "fnac")
            st.plotly_chart(fig)
            st.write("This temporal analysis shows price fluctuations for the iPhone 8 over time, providing insights into pricing trends.")

            st.subheader("Temporal Price Analysis for a Set of Products")
            fig = visualizer.analyze_temporal_for_set_of_products({
                'iphone_8 - fnac': {'brand': 'apple', 'brand_model': 'iphone 8', 'hdd': 64.0, 'color': 'prateado', 'site': 'fnac'},
                'galaxy_s8 - fnac': {'brand': 'samsung', 'brand_model': 'galaxy_s 8', 'hdd': 64.0, 'color': 'azul', 'site': 'fnac'},
                'huawei_p20 - elcorteingles': {'brand': 'huawei', 'brand_model': 'p 20', 'hdd': 128.0, 'color': 'preto', 'site': 'elcorteingles'}
            })
            st.plotly_chart(fig)
            st.write("This analysis compares the temporal price trends of multiple products across different platforms.")

            st.subheader("Discount Distribution Analysis")
            visualizer.analyze_discount_distribution()  
            st.pyplot(plt)
            st.write("This visualization highlights the distribution of discounts applied to products, showing where discounts are most common.")

            st.subheader("Temporal Price Comparison by Category")
            fig = visualizer.compare_temporal_by_category()
            st.plotly_chart(fig)
            st.write("This chart compares the temporal price trends across various product categories, helping to identify market-wide patterns.")

if __name__ == "__main__":
    main()
