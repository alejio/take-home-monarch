import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Set page config
st.set_page_config(
    page_title="Transaction Dataset Review",
    page_icon="💰",
    layout="wide"
)

# Add logo in the top right
st.image("docs/logo-color.svg", width=100)

# Title
st.title("Transaction Dataset Review")
st.markdown("**Author:** Alex Spanos  \n**Date:** 2024-12-04")
st.divider()

st.caption("""

This is a Streamlit app developed in support of Monarch's take-home assignment. 

It provides a basic exploratory data analysis of the provided datasets, focusing on high-level account (household) statistics and categorisation quality. 
""")
st.divider()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/enrichment_evidence_2024-07-08T12_24_19.655026+00_00.csv')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['household_id'] = df['household_id'].astype(str)
    df_categories = pd.read_csv("data/categories_2024-07-08T12_24_19.655026+00_00.csv")
    df_categories['household_id'] = df_categories['household_id'].astype(str)
    # Merge dataframes on category_id
    df = df.merge(df_categories, left_on=['household_id', "categorized_as"],
                  right_on=['household_id', "category_id"], how='left')
    df_dropped_duplicates = df.drop_duplicates(subset=['household_id', 'transaction_original_description', 'transaction_amount', 'transaction_date'])
    return df, df_dropped_duplicates

df, df_dropped_duplicates = load_data()

st.header("Basic descriptive analysis")
# Create columns for layout
col1, col2 = st.columns(2)

# 1. Number of households
with col1:
    num_households = df['household_id'].nunique()
    num_unique_txns = len(df_dropped_duplicates)
    
    col1a, col1b = st.columns(2)
    with col1a:
        st.metric("Total Households", num_households)
    with col1b:
        st.metric("Unique Transactions", num_unique_txns)

    # Drop duplicates based on description, amount, and date
    unique_txns = df_dropped_duplicates.groupby('household_id').agg({
        'transaction_original_description': 'count'
    }).reset_index()
    unique_txns.columns = ['household_id', 'transaction_count']
    
    # Create a Seaborn bar plot with larger figure size
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=unique_txns, x='household_id', y='transaction_count', ax=ax)
    ax.set_title("Number of Unique Transactions per Household")
    # Rotate x-axis labels and adjust their position
    plt.xticks(rotation=45, ha='right')
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    st.pyplot(fig)


with col2:
    st.subheader("Transaction history span")
    time_periods = df.groupby('household_id').agg({
        'transaction_date': ['min', 'max']
    }).reset_index()
    time_periods.columns = ['household_id', 'start_date', 'end_date']

    # Calculate duration in days
    time_periods['duration_days'] = (time_periods['end_date'] - time_periods['start_date']).dt.days

    # Add number of unique transactions
    unique_txns_count = df_dropped_duplicates.groupby('household_id').size().reset_index(name='unique_transaction_count')

    # Merge with time_periods
    time_periods = time_periods.merge(unique_txns_count, on='household_id', how='left')

    # Calculate average transactions per day
    time_periods['avg_txns_per_day'] = round(time_periods['unique_transaction_count'] / time_periods['duration_days'], 1)

    # Format the table
    time_periods['start_date'] = time_periods['start_date'].dt.strftime('%Y-%m-%d')
    time_periods['end_date'] = time_periods['end_date'].dt.strftime('%Y-%m-%d')

    # Rename columns for display
    display_df = time_periods.rename(columns={
        'household_id': 'Household ID',
        'start_date': 'First Transaction',
        'end_date': 'Last Transaction',
        'duration_days': 'Duration (days)',
        'unique_transaction_count': 'Unique Transactions',
        'avg_txns_per_day': 'Avg Transactions/Day'
    })

    st.dataframe(display_df, hide_index=True)
    st.write("Note: unusual values in transaction counts imply that at least some of the data may be synthetic.")

st.divider()

st.header("Categorisation")

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    overall_coverage = df_dropped_duplicates['system_category_name'].notna().mean() * 100
    st.subheader(f"Coverage: {overall_coverage:.2f}%")
    # Calculate overall categorisation coverage
    st.write("Coverage shows the percentage of transactions that have received a system category, regardless of accuracy.")
    # Calculate categorisation coverage per household
    categorisation = df_dropped_duplicates.groupby('household_id').agg({
        'system_category_name': lambda x: x.notna().mean() * 100
    }).reset_index()
    categorisation.columns = ['household_id', 'percent_categorised']

    # Sort by coverage ascending
    categorisation = categorisation.sort_values('percent_categorised')

    # Create a Seaborn bar plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=categorisation, x='household_id', y='percent_categorised', ax=ax)
    ax.set_title("Percentage of Categorised Transactions by Household")
    ax.set_ylim(0, 100)  # Set y-axis limits from 0 to 100
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.subheader("Common patterns in uncategorised transactions")

    st.write("Below are the most common patterns in transaction descriptions that failed to receive a category.")

    # Get uncategorised transactions
    uncategorised_txns = df_dropped_duplicates[df_dropped_duplicates['system_category_name'].isna()]

    # Add ngram size selector using radio buttons
    ngram_size = st.radio("N-gram Size", options=[1, 2, 3], horizontal=True)

    # Rest of the code remains the same
    vectoriser = CountVectorizer(ngram_range=(ngram_size, ngram_size))
    ngrams = vectoriser.fit_transform(uncategorised_txns['transaction_original_description'])

    ngram_freq = pd.DataFrame(
        ngrams.sum(axis=0).T,
        index=vectoriser.get_feature_names_out(),
        columns=['frequency']
    ).sort_values('frequency', ascending=False)

    fig = px.bar(
        ngram_freq.head(20).reset_index(),
        x='index',
        y='frequency',
        labels={'index': f'{ngram_size}-gram', 'frequency': 'Frequency'},
        title=f"Most Common {ngram_size}-grams in Uncategorised Transactions"
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)



with col2:
    st.subheader("'Accuracy' (qualitatively assessed)")
    st.write("This is a simple widget allowing to 'eyeball' the accuracy of the categorisation on a per-household basis, in the absence of ground truth. Quite a few misclassifications are evident.")

    # Add household selector before the expander
    selected_household = st.selectbox(
        "Select Household",
        options=df['household_id'].unique(),
        key='household_selector'
    )

    # Add shuffle button
    shuffle_button = st.button("Shuffle Transactions")

    # Categorised transactions explorer
    filtered_df = df_dropped_duplicates[
        (df_dropped_duplicates['household_id'] == selected_household) & 
        (df_dropped_duplicates['system_category_name'].notna())
    ][['transaction_original_description', 'system_category_name']]
    
    if shuffle_button:
        display_df = filtered_df.sample(frac=1).reset_index(drop=True)
    else:
        display_df = filtered_df
    
    st.dataframe(
        display_df, 
        hide_index=True,
        column_config={
            "transaction_original_description": "Description",
            "system_category_name": "Category"
        }
    )


