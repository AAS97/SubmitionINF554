from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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


mlp = MLPClassifier(verbose=True)

mlp.fit(X_train,y_train)

#pickle.dump(mlp, open('./ISAE_Comp/out/mlp_model.sav', 'wb'))
#print("Finished fitting, model saved to file", flush=True)





print("Getting {0} % accuracy".format(accuracy_score(y_test, prediction, normalize=False)/len(X_test)), flush=True)
print("Confusion Matrix", flush=True)
print(confusion_matrix(y_test, prediction)/len(X_test), flush=True)

#load test sample
X_validation = []
with open('./data/testing.txt', 'r') as file:
    for line in file:
        line = line.split()
        if int(line[0]) < max_node and int(line[1]) < max_node:
            X_validation.append( bow[int(line[0])].extend(bow[int(line[1])]) )

prediction = list(mlp.predict(X_validation))        

#store it in a csv file
data = zip(range(0,len(prediction)), prediction)
# Write the output in the format required by Kaggle
with open("./predictionBOW.csv","w",newline = '') as sample:
    csv_out = csv.writer(sample)
    csv_out.writerow(['id','predicted'])
    for row in data:
        csv_out.writerow(row) 

print("Prediction wrote to file", flush=True )