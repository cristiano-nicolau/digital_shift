# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import pmdarima as pm  
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')



# %% [markdown]
# ___________________________________________________________________________________________________________________________________________
# 

# %% [markdown]
# 
# Quero extrair informações em diferentes níveis:
# 
# - Categoria - ja existe
# - Marca tenho extrair do titulo
# - Modelo (com detalhes específicos por marca)
# - Site
# - Promoção
# 
# 
# Para cada nível, analisa:
# 
# Distribuição de produtos
# Preços médios
# Características específicas
# Relações entre níveis
# ....
# 
# 

# %% [markdown]
# ___________________________________________________________________________________________________________________________________________

# %%
class ProductAnalyzer:
    def __init__(self):
        self.brand_patterns = { 
            'hp': r'hp|omen by hp|pavilion', 
            'msi': r'msi', 
            'samsung': r'samsung|galaxy|note|galaxy\s*d+|galaxy\s*a\d+', 
            'apple': r'apple|iphone|ipad|macbook|mac|apple\sair', 
            'huawei': r'huawei|huawei\s*p\d+|mate\s?\d+', 
            'sony': r'sony|playstation|ps[0-9]', 
            'xiaomi': r'xiaomi|mi\s|redmi', 
            'asus': r'asus|rog|zenfone|zenbook|tuf|vivobook', 
            'lenovo': r'lenovo|thinkpad|ideapad|lenovo\slegion|yoga', 
            'dell': r'dell|inspiron|xps|latitude|alienware',
            'acer': r'acer|aspire|predator|nitro|swift', 
            'lg': r'lg|gram|v\d+|k\d+', 
            'motorola': r'motorola|motos\s|\sg\d+|\sz\d+', 
            'oneplus': r'oneplus|1\+\d+', 
            'google': r'google|pixel', 
            'microsoft': r'microsoft|surface', 
            'razer': r'razer|blade',
            'xbox' : r'xbox|series\s?x|series\s?s|xbox\sone|xbox\s360',
            'tcl' : r'tcl',
            'jbl' : r'jbl',
            'logitech' : r'logitech',
            'delonghi' : r'delonghi',
            'philips' : r'philips',
            'delta' : r'delta|delta\sq',
            'krups' : r'krups',
            'oppo' : r'oppo',
        }

        self.model_patterns = {
            'delonghi': {
            'essenza': r'essenza',
            'en': r'en\s*(\d+)',
            'inissia': r'inissia',
            'essenza_mini': r'essenza\s*mini',
            'lattissima': r'lattissima',
            },
            'delta' : {
            'q': r'qool | quick',
            },
            'krups' : {
            'essenza': r'essenza',
            'en': r'en\s*(\d+)',
            'inissia': r'inissia',
            'essenza_mini': r'essenza\s*mini',
            'lattissima': r'lattissima',
            'gusto': r'gusto',
            },
            'apple': {
            'iphone': r'iphone\s*(\d+|x\s|xr|xs|pro|max|mini|se)',
            'ipad': r'ipad\s*(\d+|pro|air|mini)',
            'macbook': r'macbook\s*(air|pro|12)',
            },
            'samsung': {
            'galaxy_s': r'galaxy\s?s(\d{1,2})',
            'galaxy_a': r'galaxy\s?a(\d{1,2})',
            },
            'huawei': {
            'p': r'p\s*(\d+)', 
            'mate': r'mate\s*(\d+)',
            'nova': r'nova\s*(\d+)',
            },
            'hp': {
            'omen': r'\s*omen\s*',
            'pavilion': r'pavilion\s*(\d+)|pavilion',
            'envy': r'envy\s*(\d+) | envy',
            'spectre': r'spectre\s*(\d+) | spectre',
            'stream' : r'stream\s*(\d+)',
            },
            'acer': {
            'aspire': r'aspire\s*(\d+)',
            'predator': r'predator\s*(\d+)',
            'nitro': r'nitro\s*(\d+)',
            'swift': r'swift\s*(\d+)',
            },
            'microsoft': {
            'surface': r'surface\s*(go|pro|x|laptop)',
            },
            'dell': {
            'inspiron': r'inspiron|inspiron\s*(\d+)',
            'xps': r'xps\s*(\d+) | xps',
            'latitude': r'latitude\s*(\d+)',
            'alienware': r'alienware\s*(\d+)',
            },
            'asus': {
            'rog': r'rog\s*(\d+) | rog',
            'zenbook': r'zenbook\s*(\d+) | zenbook',
            'vivobook': r'vivobook\s*(\d+) | vivobook',
            'tuf': r'tuf\s*(\d+) | tuf',
            },
            'lenovo': {
            'thinkpad': r'thinkpad\s*(\d+)| thinkpad',
            'ideapad': r'ideapad | ideapad\s*(\d+)',
            'legion': r'legion | legion\s*(\d+)',
            'yoga': r'yoga|yoga\s*(\d+)',
            },
            'xiaomi': {
            'mi': r'mi\s*(\d+)',
            'redmi': r'redmi\s*(\d+)',
            'poco': r'poco\s*(\d+)',
            },
            'sony': {
            'xperia': r'xperia\s*(\d+)',
            'ps4 slim': r'ps4\s*slim',
            'ps4 pro': r'ps4\s*pro',
            },
            'xbox': {
            'one': r'one\s*(x|s)?',
            'series': r'series\s*(x|s)?',
            },
        }
        
        self.color_patterns = {
            'preto': r'preto|black|negro',
            'branco': r'branco|white',
            'prata': r'prata|silver|cinzento|gray',
            'prateado' : r'prateado|silver|cinzento',
            'dourado': r'dourado|gold|amarelo|yellow',
            'azul': r'azul|blue',
            'verde': r'verde|green',
            'vermelho': r'vermelho|red',
            'rosa': r'rosa|pink',
            'roxo': r'roxo|purple',
            'laranja': r'laranja|orange',
            'castanho': r'marrom|castanho|brown',
            'cinzento': r'cinza|grey',
        }
    def process_prices(self, price_list_str):
        """Processa a lista de preços usando ast.literal_eval"""
        try:
            prices = ast.literal_eval(price_list_str)
            return prices if isinstance(prices, list) else [prices]
        except:
            return []
        
    def get_price_metrics(self, prices):
        """Calcula métricas de preço para uma lista de preços"""
        if not prices:
            return {
                'min_price': None,
                'max_price': None,
                'avg_price': None,
                'price_variation': None,
                'discount_percent': None
            }
            
        metrics = {
            'min_price': min(prices),
            'max_price': max(prices),
            'avg_price': np.mean(prices),
            'price_variation': np.std(prices) if len(prices) > 1 else 0,
            'discount_percent': ((max(prices) - min(prices)) / max(prices) * 100) if len(prices) > 1 else 0
        }
        return metrics
    
    def extract_brand(self, title):
        """Extrai a marca do título do produto."""
        title = title.lower()
        if 'gaming' in title:
            title = title.replace('gaming', '')
        for brand, pattern in self.brand_patterns.items():
            if re.search(pattern, title, re.IGNORECASE):
                return brand
        return 'Outras'
    
    def extract_model_info(self, title, brand):
        """Extrai o modelo do título do produto com base na marca."""
        title = title.lower()
        if 'gaming' in title:
            title = title.replace('gaming', '')
        model_info = {'raw_title': title, 'brand': brand}
        brand_patterns = self.model_patterns.get(brand, {})
        for model, pattern in brand_patterns.items():
            match = re.search(pattern, title)
            if match:
                sub_model = match.group(1) if match.groups() else None
                return model, sub_model, f"{model} {sub_model}" if sub_model else model
        return None, None, None  
    
    def extract_color(self, title):
        """Extrai a cor do título do produto."""
        title = title.lower()
        for color, pattern in self.color_patterns.items():
            if re.search(pattern, title):
                return color
        return None
    
    def extract_HDD_size(self, title):
        # extrai o tamanho do disco rígido 
        # nn GB ou nn TB, nnn GB ou nnn TB, nnGB ou nnTB, nnnGB ou nnnTB
        title = title.lower()
        match = re.search(r'(\d+)\s?(tb|gb)', title)
        if match:
            size = int(match.group(1))
            return size if size > 32 else None
        return None
    
    def extract_RAM_size(self, title):
        # extrai o tamanho da memória RAM 
        # nn GB
        title = title.lower()
        match = re.search(r'(\d+)\s?gb', title)
        if match:
            size = int(match.group(1))
            return size if size <=32 else None
        return None
        
    
    def compare_by_category(self, df):
        category_comparison = {
            'total_products': df.groupby('categoria').size(),
            'avg_price': df.groupby('categoria')['min_price'].mean(),
            'price_range': df.groupby('categoria').agg({
                'min_price': ['min', 'max'],
                'avg_price': 'mean'
            }),
            'brand_distribution': df.groupby(['categoria', 'brand']).size().unstack(fill_value=0)
        }
        return category_comparison
    
    def compare_by_brand(self, df):
        brand_comparison = {
            'product_count': df.groupby(['categoria', 'brand']).size(),
            'avg_price_by_category': df.groupby(['categoria', 'brand'])['avg_price'].mean(),
            'price_range_by_category': df.groupby(['categoria', 'brand']).agg({
                'min_price': ['min', 'max'],
                'max_price': ['min', 'max']
            }),
            'discount_comparison': df.groupby('brand')['discount_percent'].mean(),
            'site_presence': df.groupby(['brand', 'site']).size().unstack(fill_value=0)
        }
        return brand_comparison
    
    def compare_by_color(self, df):
        df['color'] = df['title'].apply(self.extract_color)
        color_comparison = {
            'total_products': df.groupby('color').size(),
            'avg_price': df.groupby('color')['min_price'].mean(),
            'price_range': df.groupby('color').agg({
                'min_price': ['min', 'max'],
                'avg_price': 'mean'
            }),
            'brand_distribution': df.groupby(['color', 'brand']).size().unstack(fill_value=0)
        }
        return color_comparison
    
    def compare_models(self, df):
        model_comparison = {
            'total_products': df.groupby('brand_model').size(),
            'avg_price': df.groupby('brand_model')['min_price'].mean(),
            'price_range': df.groupby('brand_model').agg({
                'min_price': ['min', 'max'],
                'avg_price': 'mean'
            }),
            'brand_distribution': df.groupby(['brand_model', 'brand']).size().unstack(fill_value=0)
        }
        return model_comparison
    

    def compare_by_price_range(self, df):
        bins = [0, 500, 1000, 1500, 2000, 3000, 5000, float('inf')]
        labels = ['0-500', '500-1000', '1000-1500', '1500-2000', '2000-3000', '3000-5000', '5000+']
        df['price_range'] = pd.cut(df['min_price'], bins=bins, labels=labels)
        price_range_comparison = {
            'total_products': df.groupby('price_range').size(),
            'avg_price': df.groupby('price_range')['min_price'].mean(),
            'brand_distribution': df.groupby(['price_range', 'brand']).size().unstack(fill_value=0)
        }
        return price_range_comparison
    
    def compare_temporal_analysis(self, df):
        """
        Realiza uma análise temporal dos preços por produto e site.
        """
        if 'date' not in df.columns or df['date'].isna().all():
            raise ValueError("O dataset não possui informações de data suficientes para a análise temporal.")
        
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        if df['date'].isna().all():
            raise ValueError("Não foi possível converter as datas para o formato datetime.")
        
        required_columns = ['site', 'brand', 'title', 'min_price', 'discount_percent']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"A coluna obrigatória '{col}' não está presente no dataset.")
        
        temporal_data = df.groupby(['site', 'brand','brand_model' ,'color', 'HDD_size', 'date']).agg({
            'min_price': 'mean', 
            'discount_percent': 'mean' 
        }).reset_index()
        
        temporal_data = temporal_data.sort_values(by=['site', 'brand_model', 'date'])
        
        temporal_data['price_change_percent'] = temporal_data.groupby(['site','brand', 'brand_model', 'color', 'HDD_size'])['min_price'].pct_change() * 100
        
        summary = temporal_data.groupby(['site','brand', 'brand_model', 'color', 'HDD_size']).agg({
            'price_change_percent': ['mean', 'std'],  
            'min_price': ['min', 'max', 'mean'], 
            'discount_percent': 'mean'  
        }).reset_index()

        # Ajustar os nomes das colunas no resumo
        summary.columns = ['site','brand', 'brand_model', 'color', 'HDD_size', 'mean_price_change', 'std_price_change', 
                        'min_price', 'max_price', 'avg_price', 'avg_discount']
        
        return temporal_data, summary
    
    def analyze_dataset(self, df):
        """
        Processa um dataset de produtos e realiza diversas análises, incluindo comparações
        por categoria, marca, cor, site e data, além de detalhes do modelo.
        """
        df = df.copy()
        
        # Processar preços
        df['price_list'] = df['extractedData'].apply(self.process_prices)
        
        # Calcular métricas de preço
        price_metrics = df['price_list'].apply(self.get_price_metrics)
        price_metrics_df = pd.DataFrame(price_metrics.tolist())
        df = pd.concat([df, price_metrics_df], axis=1)

        df['brand'] = df['title'].apply(self.extract_brand)
        df[['model', 'sub_model', 'brand_model']] = pd.DataFrame(
            df.apply(lambda x: self.extract_model_info(x['title'], x['brand']), axis=1).tolist(),
            index=df.index
        )
        df['color'] = df['title'].apply(self.extract_color)
        df['HDD_size'] = df['title'].apply(self.extract_HDD_size)
        df['RAM_size'] = df['title'].apply(self.extract_RAM_size)

        # Extrair detalhes do modelo
        df['model_details'] = df.apply(
        lambda x: self.extract_model_info(x['title'], x['brand']), axis=1
        )

        # Comparações agregadas
        comparisons = {
            'by_category': self.compare_by_category(df),
            'by_brand': self.compare_by_brand(df),
            'by_color': self.compare_by_color(df),
            'by_site': df.groupby('site').agg({
                'min_price': ['min', 'max', 'mean'],
                'discount_percent': 'mean',
            }),
            'by_date': df.groupby('date').agg({
                'min_price': ['min', 'max', 'mean'],
                'discount_percent': 'mean',
            }),
            'by_model': self.compare_models(df),
            'by_HDD': df.groupby('HDD_size').agg({
                'min_price': ['min', 'max', 'mean'],
                'discount_percent': 'mean',
            }),
            'by_RAM': df.groupby('RAM_size').agg({
                'min_price': ['min', 'max', 'mean'],
                'discount_percent': 'mean',
            }),
            'by_price_range': self.compare_by_price_range(df),
        }

        return df, comparisons




# %%
df = pd.read_csv("all_prices_extracted_cleaned_categorized.csv")
analyzer = ProductAnalyzer()
df_analyzed, insights = analyzer.analyze_dataset(df)
df_analyzed.to_csv("all_prices_extracted_cleaned_categorized_analyzed.csv", index=False)

# %%
class ProductVisualizer:
    def __init__(self, df_analyzed, insights):
        self.df = df_analyzed
        self.insights = insights
        self.analyser = ProductAnalyzer()
    
    def plot_category_distribution(self):
        """Distribuição de produtos por categoria baseada nos dados filtrados"""
        category_counts = self.df.groupby('categoria').size().reset_index(name='total_products')
        category_counts.columns = ['categoria', 'total_products']
        
        fig = px.bar(
            category_counts,
            x='categoria',
            y='total_products',
            labels={'categoria': 'Categoria', 'total_products': 'Número de Produtos'},
            color='categoria'
        )
        fig.update_layout(
            xaxis_title='Categoria',
            yaxis_title='Número de Produtos',
            legend_title='Categoria'
        )
        return fig
    
    def plot_brand_market_share(self):
        """Market share por categoria"""
        brand_distribution = self.df.groupby(['categoria', 'brand']).size().reset_index(name='product_count')
        brand_distribution = brand_distribution.pivot(index='categoria', columns='brand', values='product_count').fillna(0)

        fig = px.bar(
            brand_distribution,
            labels={'value': 'Número de Produtos', 'index': 'Categoria'},
            barmode='stack',
            color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set3
        )
        fig.update_layout(
            xaxis_title='Categoria',
            yaxis_title='Número de Produtos',
            legend_title='Marca'
        )
        return fig
    def plot_price_comparison(self):
        """Comparação de preços por marca e categoria"""

        avg_prices = self.df.groupby(['categoria', 'brand'])['min_price'].mean()

        
        fig = px.bar(
            avg_prices.unstack(),
            labels={'value': 'Preço Médio', 'index': 'Categoria - Marca'},
            barmode='group'
        )
        fig.update_layout(
            xaxis_title='Categoria - Marca',
            yaxis_title='Preço Médio',
            legend_title='Marca'
        )
        return fig

    def plot_discount_analysis(self):
        """Análise de descontos por marca"""
        discount_comparison = self.insights['by_brand']['discount_comparison'].reset_index()
        discount_comparison.columns = ['brand', 'discount_percent']
        
        fig = px.bar(
            discount_comparison,
            x='brand',
            y='discount_percent',
            labels={'brand': 'Marca', 'discount_percent': 'Desconto Médio (%)'},
            color='brand',
            color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set3
        )
        fig.update_layout(
            title='Percentual de Desconto Médio por Marca',
            xaxis_title='Marca',
            yaxis_title='Desconto Médio (%)',
            legend_title='Marca'
        )
        return fig
    
    def plot_site_brand_presence(self):
        """presenca de marcas por site"""
        site_brand_presence = self.df.groupby(['brand', 'site']).size().unstack(fill_value=0)
        print(site_brand_presence)
        fig = px.imshow(
            site_brand_presence,
            labels={"x": "site", "y": "brand", "color": "Presence"},
            text_auto=True,
            color_continuous_scale='Peach',
            aspect="auto"
        )
        
        fig.update_layout(
            xaxis_title="Sites",
            yaxis_title="Marcas",
            autosize=True
        )
        return fig

    def plot_model_series_distribution_iphone8(self):
        """distribuicao do numero de produtos por modelo"""

        model_distribution = self.df.groupby(['brand', 'brand_model']).size().reset_index(name='total_products')

        fig = px.bar(
            model_distribution,
            x='brand_model',
            y='total_products',
            color='brand',
            labels={'total_products': 'Número de Produtos', 'brand_model': 'Modelo', 'brand': 'Marca'},
            color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            xaxis_title='Modelo',
            yaxis_title='Número de Produtos',
            barmode='group',
            legend_title='Marca'
        )
        
        return fig
        
    
    def analyze_discount_distribution(self):
        """
        Analisa a distribuição de descontos em faixas de desconto utilizando Plotly.
        """
        bins = [0, 10, 20, 30, 40, 50, 100]
        labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+']

        self.df['discount_range'] = pd.cut(self.df['discount_percent'], bins=bins, labels=labels)

        discount_distribution = self.df['discount_range'].value_counts().reset_index()
        discount_distribution.columns = ['discount_range', 'count']

        fig = px.bar(
            discount_distribution,
            x='discount_range',
            y='count',
            color='count',
            color_continuous_scale='Peach',
            text_auto=True,
            title="Distribuição de Produtos por Faixa de Desconto"
        )
        fig.update_layout(
            xaxis_title="Faixa de Desconto (%)",
            yaxis_title="Quantidade de Produtos",
            template="plotly_white",
            showlegend=False,
            autosize=True
        )

        return fig
    
    def compare_temporal_by_category(self):
        """
        comparacao temporal dos precos por categoria
        """
        temporal_data = self.df.groupby(['categoria', 'date']).agg({
            'min_price': 'mean'
        }).reset_index()
        
        temporal_data['date'] = pd.to_datetime(temporal_data['date'], format='%Y%m%d')
        
        fig = px.line(
            temporal_data,
            x='date',
            y='min_price',
            color='categoria',
            title="Média de Preços por Categorias ao Longo do Tempo",
            labels={'date': 'Data', 'min_price': 'Preço Médio (€)', 'categoria': 'Categoria'},
            color_discrete_sequence=px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            xaxis_title='Data',
            yaxis_title='Preço Médio (€)',
            legend_title='Categoria',
            hovermode='x unified'
        )
        
        return fig
        


    def analyze_price_range(_df_analyzed, price_min, price_max):
        """
        Realiza análise de produtos em uma faixa de preço específica.
        """
        # Filtrar produtos pela faixa de preço
        filtered_df = df_analyzed[(df_analyzed['min_price'] >= price_min) & (df_analyzed['min_price'] <= price_max)]
        
        if filtered_df.empty:
            return f"Nenhum produto encontrado na faixa de preço €{price_min} - €{price_max}."
        
        # Comparar produtos mais baratos e mais caros
        cheapest_product = filtered_df.loc[filtered_df['min_price'].idxmin()]
        most_expensive_product = filtered_df.loc[filtered_df['min_price'].idxmax()]
        
        # Visualizar distribuição de preços na faixa
        plt.figure(figsize=(10, 6))
        plt.hist(filtered_df['min_price'], bins=20, color='skyblue', edgecolor='black')
        plt.title(f"Distribuição de Preços na Faixa: €{price_min} - €{price_max}")
        plt.xlabel("Preço (€)")
        plt.ylabel("Quantidade de Produtos")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        return plt
    
    def price_discount_correlation(self):
        scatter_data = self.df.dropna(subset=['min_price', 'discount_percent', 'site', 'categoria'])
        scatter_data = scatter_data[scatter_data['discount_percent'] > 0]

        scatter_data['price_with_discount'] = scatter_data['min_price'] * (1 - scatter_data['discount_percent'] / 100)

        fig = px.scatter(
            scatter_data,
            x='min_price',
            y='discount_percent',
            color='categoria',
            size='discount_percent', 
            hover_data=['price_with_discount','title', 'site'], 
            labels={"min_price": "Preço Mínimo (€)", "discount_percent": "Desconto (%)", "price_with_discount": "Preço com Desconto (€)"}
        )

        fig.update_layout(
            xaxis_title="Preço Mínimo (€)",
            yaxis_title="Desconto (%)",
            legend_title="Categorias",
            hovermode="closest"
        )

        return fig

    def number_of_products_by_site(self):
        site_counts = self.df.groupby('site').size().reset_index(name='total_products')
        site_counts.columns = ['site', 'total_products']

        fig = px.pie(
            site_counts,
            values='total_products',
            names='site',
            title='Number of Products by Site',
            labels={'total_products': 'Número de Produtos', 'site': 'Site'}
        )
        return fig
    
    def compare_promotions(self):
        promotion_counts = self.df.groupby('promocao').size().reset_index(name='total_products')
        promotion_counts.columns = ['promocao', 'total_products']
        
        fig = px.pie(
            promotion_counts,
            values='total_products',
            names='promocao',
            title='Number of Products with Promotions',
            labels={'total_products': 'Número de Produtos', 'promocao': 'Promoção'}
        )
        return fig
    

    def products_most_discounted(self, num_products, selected_sites):
        products_discounts = self.df[['site', 'title', 'discount_percent']].dropna()
        products_discounts = products_discounts[products_discounts['site'].isin(selected_sites)]
        top_discounts = products_discounts.sort_values(by='discount_percent', ascending=False).head(num_products)
        fig = px.bar(
            top_discounts,
            x='discount_percent',
            y='title',
            color='site',
            orientation='h',
            text_auto=True,
            labels={"discount_percent": "Desconto (%)", "title": "Produto"}
        )
        fig.update_yaxes(categoryorder='total ascending')
        fig.update_layout(
            barmode="stack",
            xaxis_title="Percentual de Desconto",
            yaxis_title="Produtos",
            showlegend=True
        )

        return fig  
class TimeSeriesAnalyzer:
    def __init__(self, df):
        self.df = df
        self.df['date'] = pd.to_datetime(self.df['date'], format='%Y%m%d')
        self.forecasts = {}

    def prepare_time_series(self, brand=None, model=None, color=None, category=None, brand_model=None):
        filtered_df = self.df.copy()

        if brand:
            filtered_df = filtered_df[filtered_df['brand'] == brand]
        if model:
            filtered_df = filtered_df[filtered_df['model'] == model]
        if color:
            filtered_df = filtered_df[filtered_df['color'] == color]
        if category:
            filtered_df = filtered_df[filtered_df['categoria'] == category]
        if brand_model:
            filtered_df = filtered_df[filtered_df['brand_model'] == brand_model]

        time_series = filtered_df.groupby('date')['min_price'].mean().reset_index()
        time_series.set_index('date', inplace=True)

        if time_series.empty or len(time_series) < 3:
            return None

        return time_series

    def forecast_time_series_sarima(self, time_series, forecast_periods):
        if len(time_series) < 2:
            return None

        prices = time_series['min_price'].values

        try:
            sarima_model = pm.auto_arima(
                prices,
                seasonal=False,
                suppress_warnings=True,
                stepwise=True,
            )

            sarima_forecast, conf_int = sarima_model.predict(n_periods=forecast_periods, return_conf_int=True)

            # adiciono lhe uma multiplicaçao para ajustar a tendência decrescente
            decline_factor = 0.975
            sarima_forecast = sarima_forecast * (decline_factor ** np.arange(0, forecast_periods))

            return {
                'forecast': sarima_forecast,
                'conf_int': conf_int
            }
        except Exception:
            return None

    def analyze_product(self, brand, model, color, category, brand_model):
        time_series = self.prepare_time_series(
            brand=brand, model=model, color=color, category=category, brand_model=brand_model
        )

        if time_series is None:
            return None

        last_date = time_series.index[-1]
        end_of_2024 = pd.Timestamp('2024-12-31')
        forecast_periods = (end_of_2024.year - last_date.year) * 12 + (end_of_2024.month - last_date.month)

        if forecast_periods <= 0:
            return None

        product_forecasts = self.forecast_time_series_sarima(time_series, forecast_periods)

        if product_forecasts:
            return {
                'time_series': time_series,
                'forecasts': product_forecasts
            }
        return None


class ProductPricePrediction:
    def __init__(self, df):
        self.df = df

        self.numeric_features = ['HDD_size', 'RAM_size']
        self.categorical_features = ['site', 'categoria', 'brand', 'model', 'color']

        # processar promocao em binario
        self.df['promocao'] = self.df['promocao'].apply(lambda x: 0 if x == 'Sem Promocao' else 1)

        # adicionar 'promocao' as features numéricas após a transformação
        self.numeric_features.append('promocao')

        # scaler para normalizar os valores numéricos, one-hot encoder para as features categóricas
        # impute com os valores mais frequentes para as features categoricas e com a media para as features numericas
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), self.numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), self.categorical_features)
            ]
        )

        self.price_model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

    def prepare_data(self, price_column='max_price'):
        """Prepara os dados para treino."""
        # dar input de valores medianos para os valores nulos
        self.df[price_column] = self.df[price_column].fillna(self.df[price_column].median())

        # dividir em features e target, o target é o max preço
        X = self.df.drop([price_column], axis=1)
        y = self.df[price_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_model(self):
        """Treina o modelo para prever os preços."""
        X_train, X_test, y_train, y_test = self.prepare_data()

        self.price_model.fit(X_train, y_train)
        predictions = self.price_model.predict(X_test)

    def predict_price(self, product_data):
        """Faz a previsão do preço com base nas características do produto."""
        input_df = pd.DataFrame([product_data])
        predicted_price = self.price_model.predict(input_df)[0]
        return predicted_price

