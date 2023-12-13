import streamlit as st
from pydataset import data

st.title('pydataset')

select_data = st.sidebar.selectbox('select a dataset', data().dataset_id)

st.header('Datasets')
st.subheader('List of dataset');
with st.expander('Show List of dataset'):
     st.write(data())

st.subheader(f'select data ('{select_data}')')
st.write(data(select_data))
