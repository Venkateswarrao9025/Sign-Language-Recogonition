import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dt_dict = pickle.load(open('./data.pickle', 'rb'))

dt = dt_dict['data']
lbl = dt_dict['labels']

max_len = max(len(i) for i in dt)

padded_data = [np.pad(i, (0, max_len - len(i))) for i in dt]

flattened_data = [np.array(i).flatten() for i in padded_data]

a_training, a_testing, b_training, b_testing = train_test_split(flattened_data, lbl, test_size=0.2, shuffle=True, stratify=lbl)

model = RandomForestClassifier()

model.fit(a_training, b_training)

b_predict = model.predict(a_testing)

scr = accuracy_score(b_predict, b_testing)

print(scr * 100,'% of samples were appropriately categorized...!!!')

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
