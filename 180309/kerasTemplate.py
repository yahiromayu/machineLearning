n_in = 784
n_hiddens = [200, 200]
n_out = 10
activation = "relu"
p_keep = 0.5

model = Sequential()
for i, input_dim in enumerate(([n_in] + n_hiddens)[:-1]):
	model.add(Dense(n_hiddens[i],input_dim=input_dim))
	model.add(Activation(activation))
	model.add(Dropout(p_keep))

model.add(Dense(n_out))
model.add(Activation("softmax"))