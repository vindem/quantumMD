import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from math import sqrt
import pandas as pd
import random


input = pd.read_csv('full-quantum-dataset.csv')
cols = ['QUBITS','PQC', 'REPS', 'OPTIMIZER', 'ENTANGLEMENT']
#handle categorical data
lb_make = LabelEncoder()
input['PQC'] = lb_make.fit_transform(input['PQC'])
pqc_map = dict(zip(lb_make.classes_, lb_make.transform(lb_make.classes_)))
input['OPTIMIZER'] = lb_make.fit_transform(input['OPTIMIZER'])
opt_map = dict(zip(lb_make.classes_, lb_make.transform(lb_make.classes_)))
input['ENTANGLEMENT'] = lb_make.fit_transform(input['ENTANGLEMENT'])
ent_map = dict(zip(lb_make.classes_, lb_make.transform(lb_make.classes_)))
#end of categorical data handling

#model fitting
y = input["SQUARED ERROR"]
X = input[cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42, shuffle=True)

model = SVR()
model.fit(X_train.values, y_train.values)
#evaluate model
predictions = model.predict(X_test.values)
print("Results of model fitting: " + str(sqrt(mean_squared_error(y_test, predictions))/(max(y_test)-min(y_test))))


def target_function(in_qubits, pqc, rep, optimizer, ent):
    pqc_transform = pqc
    optimizer_transform = optimizer
    #pqc_transform = pqc_map[pqc]
    #optimizer_transform = opt_map[optimizer]
    X = np.array([in_qubits, pqc_transform, rep, optimizer_transform, ent])
    return model.predict(X.reshape(1, -1))

def hyp_opt_bruteforce(n_qubit):
    min = float('inf')
    hyps = None
    for pqc in pqc_map.values():
        for rep in range(1,6):
            for opt in opt_map.values():
                for ent in ent_map.values():
                    val = target_function(n_qubit, pqc, rep, opt, ent)
                    if val < min:
                        hyps = [list(pqc_map.keys())[list(pqc_map.values()).index(pqc)],
                                rep,
                                list(opt_map.keys())[list(opt_map.values()).index(opt)],
                                list(ent_map.keys())[list(ent_map.values()).index(ent)]]
                        min = val
    return hyps

def rand_hyp_opt(n_qubit):
    pqc = random.choice(list(pqc_map.values()))
    rep = random.choice(range(1,6))
    opt = random.choice(list(opt_map.values()))
    ent = random.choice(list(ent_map.values()))
    return [list(pqc_map.keys())[list(pqc_map.values()).index(pqc)],
            rep,
            list(opt_map.keys())[list(opt_map.values()).index(opt)],
            list(ent_map.keys())[list(ent_map.values()).index(ent)]]


print("Hyperparameter search")
print(hyp_opt_bruteforce(1))
print(hyp_opt_bruteforce(2))
print(hyp_opt_bruteforce(3))
print(hyp_opt_bruteforce(4))
print(hyp_opt_bruteforce(5))
print("Random Selection")
print(rand_hyp_opt(1))
print(rand_hyp_opt(2))
print(rand_hyp_opt(3))
print(rand_hyp_opt(4))
print(rand_hyp_opt(5))
