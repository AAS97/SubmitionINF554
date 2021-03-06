from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import json
import pickle

with open('./ISAE_Comp/out/BOW_pca.json', 'r') as file:
    bow = json.load(file)
bow = {int(k):v for k,v in bow.items()}

X = []
y = []

with open('./data/training.txt', 'r') as file:
    for line in file:
        line = line.split()
        i = int(line[0])
        j = int(line[1])
        feat = bow[i] + bow[j]
        X.append( feat )
        y.append(line[2])

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#print("Training set legnth : {0}   Testing set length : {1}".format(len(X_train), len(X_test)), flush=True)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

mlp = MLPClassifier(verbose=True)

mlp.fit(X,y)

#pickle.dump(mlp, open('./ISAE_Comp/out/mlp_model.sav', 'wb'))
print("Finished fitting, model saved to file", flush=True)

#load test sample
X_validation = []
with open('./data/testing.txt', 'r') as file:
    for line in file:
        line = line.split()
        X_validation.append( bow[int(line[0])].extend(bow[int(line[1])]) )

X_validation = scaler.transform(X_validation)
prediction = list(mlp.predict(X_validation))        

#store it in a csv file
# Write the output in the format required by Kaggle
with open("./predictionBOW.csv","w",newline = '') as sample:
    csv_out = csv.writer(sample)
    csv_out.writerow(['id','predicted'])
    for row in zip(range(0,len(prediction)), prediction):
        csv_out.writerow(row) 

print("Prediction wrote to file", flush=True )
