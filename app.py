import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from text_model import ProductAnalyzer, ProductVisualizer  # Importe o seu módulo que contém as classes necessárias

def main():

    st.title("Análise de Produtos e Preços")

    uploaded_file = ("all_prices_extracted_cleaned_categorized.csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        analyzer = ProductAnalyzer()
        df_analyzed, insights = analyzer.analyze_dataset(df)

        # Criando uma instância do visualizador
        visualizer = ProductVisualizer(df_analyzed, insights)
 
        f = pd.read_csv('all_prices_extracted_cleaned_categorized_analyzed.csv')
        # Sites com maior diversidade de marcas
        site_brand_counts = f.groupby('site')['brand'].nunique().sort_values(ascending=False)
        st.write("Sites com Mais Marcas Únicas:")
        st.dataframe(site_brand_counts)
        
        # Marcas presentes em mais sites
        brand_site_counts = f.groupby('brand')['site'].nunique().sort_values(ascending=False)
        st.write("Marcas Presentes em Mais Sites:")
        st.dataframe(brand_site_counts)
        
        # Marcas associadas a cada site
        st.subheader("Marcas Associadas a Cada Site")
        site_brands = f.groupby('site')['brand'].apply(lambda x: list(set(x))).reset_index(name='brands')
        st.write(site_brands)
        
        # Modelos associados a cada marca
        st.subheader("Modelos Associados a Cada Marca")
        brand_models = f.groupby('brand')['brand_model'].apply(lambda x: list(set(x))).reset_index(name='models')
        st.write(brand_models)
        
        # Categoria associadas a cada modelo
        st.subheader("Categoria Associadas a Cada Modelo")
        model_categories = f.groupby('model')['categoria'].apply(lambda x: list(set(x))).reset_index(name='categories')
        st.write(model_categories)


        # Mostrar gráficos no Streamlit

        st.subheader("Distribuição de Produtos por Categoria")
        fig = visualizer.plot_category_distribution()  # Método Plotly
        st.plotly_chart(fig)

        st.subheader("Market Share por Marca e Categoria")
        fig = visualizer.plot_brand_market_share()  # Método Plotly
        st.plotly_chart(fig)

        st.subheader("Comparação de Preços por Marca e Categoria")
        fig = visualizer.plot_price_comparison()  # Método Plotly
        st.plotly_chart(fig)

        st.subheader("Análise de Descontos por Marca")
        visualizer.plot_discount_analysis()  # Método Matplotlib
        st.pyplot(plt)

        st.subheader("Presença de Marcas por Site")
        visualizer.plot_site_brand_presence()  # Método Matplotlib
        st.pyplot(plt)

        st.subheader("Distribuição de Modelos de Produtos")
        fig = visualizer.plot_model_series_distribution_iphone8()  # Método Plotly
        st.plotly_chart(fig)

        st.subheader("Análise Temporal de Preços para o iPhone 8")
        fig = visualizer.analyze_temporal_for_product("apple", "iphone 8", 64.0, "prateado", "fnac")
        st.plotly_chart(fig)

        st.subheader("Análise Temporal de Preços para um Conjunto de Produtos")
        fig = visualizer.analyze_temporal_for_set_of_products({
            'iphone_8 - fnac': {'brand': 'apple', 'brand_model': 'iphone 8', 'hdd': 64.0, 'color': 'prateado', 'site': 'fnac'},
            'galaxy_s8 - fnac': {'brand': 'samsung', 'brand_model': 'galaxy_s 8', 'hdd': 64.0, 'color': 'azul', 'site': 'fnac'},
            'huawei_p20 - elcorteingles' : {'brand': 'huawei', 'brand_model': 'p 20', 'hdd': 128.0, 'color': 'preto', 'site': 'elcorteingles'}
        })
        st.plotly_chart(fig)

        st.subheader("Análise de Distribuição de Descontos")
        visualizer.analyze_discount_distribution()  # Método Matplotlib
        st.pyplot(plt)

        st.subheader("Comparação Temporal de Preços por Categoria")
        fig = visualizer.compare_temporal_by_category()
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
