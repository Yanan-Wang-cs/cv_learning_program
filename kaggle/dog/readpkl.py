import pickle
f = open('data.pkl', 'rb')
data = pickle.load(f)
print(data)