import streamlit as st 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
st.title("Penguine Classifier")
st.write(
    """This app uses 6 inputs to predict the species of penguin using
    a model built on Palmer Penguins dataset. Use the form below
    to get started!
    """
)

penguine_file = st.file_uploader("Upload your own penguine data")

if penguine_file is None:
    rf_pickle = open("random_forest_penguine.pickle",'rb')
    map_pickle = open("output_penguine.pickle",'rb')
    rfc = pickle.load(rf_pickle)
    unique_penguine_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()
    st.stop()

else:
    penguine_df = pd.read_csv(penguine_file)
    penguine_df = penguine_df.dropna()
    output = penguine_df['species']
    features = penguine_df[['island','bill_length_mm','bill_depth_mm', 'flipper_length_mm', 'body_mass_g','sex']]
    features = pd.get_dummies(features)
    output, unique_penguine_mapping = pd.factorize(output)
    x_train,x_test,y_train,y_test = train_test_split(features,output,test_size=.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train.values,y_train)
    y_pred = rfc.predict(x_test.values)
    score = round(accuracy_score(y_test,y_pred),2)
    st.write(
        f"""We trained a Random Forest on these Data,
        it has a score of {score}! use the input below to try out the model."""
    )
with st.form('user_inputs'):
    island = st.selectbox("Penguine Island", options=['Biscoe','Dream','Torgerson'])
    sex = st.selectbox("Sex", options=["Female", 'Male'])
    bill_length = st.number_input("Bill Length (mm)",min_value=0)
    bill_depth = st.number_input("Bill Depth (mm)",min_value=0)
    flipper_length = st.number_input("Flipper Length (mm)",min_value=0)
    body_mass = st.number_input("Body Mass (g)",min_value=0)
    st.form_submit_button()
    island_biscoe, island_dream, island_torgerson = 0,0,0
    if island == 'Biscoe':
        island_biscoe = 1
    elif island == 'Dream':
        island_dream = 1
    elif island == 'Torgerson':
        island_torgerson=1
    sex_female, sex_male = 0,0
    if sex == 'Female':
        sex_female =1
    elif sex == 'Male':
        sex_male = 1
        
    new_prediction = rfc.predict(
        [
            [
                bill_length,
                bill_depth,
                flipper_length,
                body_mass,
                island_biscoe,
                island_dream,
                island_torgerson,
                sex_female,
                sex_male
            ]
        ]
    )
    prediction_species = unique_penguine_mapping[new_prediction][0]
    st.write(f"We predict your penguin is of the {prediction_species} species!")
    st.write("""We used a machine learning (Random Forest Classifier)
             model to predict the species, the features used in the prediction are ranky by 
             relative importance below
             """)
    st.image('feature_importance.png')
    st.write("""Below are the histograms for each continuous varible seperated by penguin species.
             The Vertical line represents your inputted value.
             """)
    fig,ax = plt.subplots()
    ax = sns.displot(x=penguine_df['bill_length_mm'],hue=penguine_df['species'])
    plt.axvline(bill_length)
    st.pyplot(ax)
    fig,ax = plt.subplots()
    ax = sns.displot(x=penguine_df['bill_depth_mm'],hue=penguine_df['species'])
    plt.axvline(bill_depth)
    st.pyplot(ax)
    fig,ax = plt.subplots()
    ax = sns.displot(x=penguine_df['flipper_length_mm'],hue=penguine_df['species'])
    plt.axvline(flipper_length)
    st.pyplot(ax)