import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
penguins_df = pd.read_csv("penguins.csv")
penguins_df.dropna(inplace=True)
output = penguins_df['species']
features = penguins_df[['island','bill_length_mm','bill_depth_mm','body_mass_g','flipper_length_mm','sex']]
features = pd.get_dummies(features)
output, uniques = pd.factorize(output) #encode species -> just like defining a dictionary and replacing species column
x_train,x_test,y_train,y_test = train_test_split(features,output,test_size=.8)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test.values)
score = accuracy_score(y_test,y_pred)
print("Our accuracy score for this model is {}".format(score))
rfc_pickle = open('random_forest_penguine.pickle','wb')
pickle.dump(rfc,rfc_pickle)
rfc_pickle.close()
output_pickle = open("output_penguine.pickle",'wb')
pickle.dump(uniques,output_pickle)
output_pickle.close()
fig,ax = plt.subplots()
ax = sns.barplot(x=rfc.feature_importances_, y=features.columns)
plt.title("Which features are the most important for the species prediction?")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
fig.savefig("feature_importance.png")