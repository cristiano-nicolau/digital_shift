
# Digital Shift: The Evolution of Products and Platforms in Portuguese E-commerce

## Introduction:

E-commerce has become a major player in the retail landscape, especially in sectors like electronics, where online shopping has seen significant growth.

In recent years, the competition among retailers in Portugal as Worten, Fnac, Rádio Popular, El Corte Inglés, Pc Diga and Staples has intensified. This competition has driven changes in both the types of products offered and the functionalities of their respective platforms, shaping the way consumers interact with online shopping.

By analyzing historical data from Arquivo.pt, it is possible to explore the evolution of these e-commerce platforms, providing insights into how product offerings and platform features have developed over time.

## Context

The Arquivo.pt platform stores archived web pages dating back to 1996, making it an invaluable resource for examining the historical trajectory of e-commerce in Portugal.

For this project, Arquivo.pt will be the primary data source, allowing us to analyze past versions of e-commerce websites for major electronics retailers such as Worten, Fnac, Rádio Popular, El Corte Inglés, Pc Diga and Staples.

By examining archived versions of these websites, we aim to track the changes in product offerings (with a focus on electronic devices like smartphones, laptops, and televisions) and investigate how the functionalities of these platforms have evolved to improve the customer experience.

## Goals

The main objectives of this project are:

- **Analyze the evolution of electronic product offerings over the years**, identifying key trends in product categories (e.g., smartphones, televisions, laptops), and examining which products gained popularity or disappeared from the market.
- **Compare specific devices**, such as various models of the iPhone, in terms of pricing and features across multiple retailers, highlighting how these factors have shifted over time.
- **Evaluate the evolution of e-commerce platform functionalities**, particularly search mechanisms, filters, and personalization tools, to understand how they improved the user experience.
- **Examine price trends and promotional strategies** used by different retailers during key sales periods, such as Black Friday or Christmas, to identify which platforms offer the best deals.

## Methodology

The methodology for this project involves the following steps:

1. **Data Collection**: Use the Arquivo.pt API to retrieve archived versions of e-commerce websites for major electronics retailers in Portugal.
2. **Data Preprocessing**: Clean and structure the title and clean the duplicated data.
3. **Data Categurization**: Categorize the data into product categories, brands, and other relevant attributes to facilitate analysis.
4. **Data Analysis**: Analyze the data to identify trends in product categories, pricing, and platform functionalities over time.
5. **Data Visualization**: Create visualizations (e.g., charts, graphs) to present the findings and insights from the analysis.
6. **Data Prediction**: Predict the prices of the products in the future based on the historical data.


## Packages

- pandas
- numpy
- matplotlib
- seaborn
- requests
- beautifulsoup4

## Directory Structure && Files

- `presentation` : Contains the presentation slides for the project.
- `data/`: Contains the data files used in the project.
- `links_extraction.ipynb`: Jupyter notebook for extracting the links from the Arquivo.pt API.
- `initial_data_analysis.ipynb`: Jupyter notebook for the initial data analysis.
- `web_scrapper.ipynb`: Jupyter notebook for web scraping the data from Arquivo.pt links.
- `data_processor.ipynb`: Jupyter notebook for processing the data and clean the data.
- `categorization_model.ipynb`: Jupyter notebook for categorizing the data, in categories, brands, and other relevant attributes.
- `ml.ipynb`: Jupyter notebook for predicting the prices of the products in the future based on the historical data.
- `app.py`: Streamlit application for the project.

## How to run the project

1. Clone the repository:

```bash
git clone git@github.com:cristiano-nicolau/digital_shift.git
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

4. Open the browser and go to the following URL:

```
http://localhost:8501
```

If you want you can see the streamlit application running [here](https://digital-shift.streamlit.app)


## Team Members
- [Cristiano Nicolau, 108536](mailto:cristianonicolau@ua.pt)

