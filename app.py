import os
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from text_model import ProductAnalyzer, ProductVisualizer  

def main():

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
        
        st.subheader("Brands Associated with Each Site")
        site_brands = f.groupby('site')['brand'].apply(lambda x: list(set(x))).reset_index(name='brands')
        st.write(site_brands)
        
        st.subheader("Models Associated with Each Brand")
        brand_models = f.groupby('brand')['brand_model'].apply(lambda x: list(set(x))).reset_index(name='models')
        st.write(brand_models)
        
        st.subheader("Categories Associated with Each Model")
        model_categories = f.groupby('model')['categoria'].apply(lambda x: list(set(x))).reset_index(name='categories')
        st.write(model_categories)


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
