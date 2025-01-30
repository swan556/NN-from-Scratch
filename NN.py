import numpy as np
import classes_and_functions as cf
import matplotlib.pyplot as plt

data = cf.get_data()
X, y = data.initialize("data.json")
hidden_layer_1 = cf.layer(63,64)
output_layer = cf.layer(64,4)

relu_activation = cf.relu()
softmax_activation = cf.softmax()
loss_activation = cf.softmax_xentropy()
optimizer = cf.optimiser_SGD()

loss_list = []
accuracy_list = []

for epoch in range(1000):    
    #forward pass
    hidden_layer_1.forward(X)
    relu_activation.forward(hidden_layer_1.z)

    output_layer.forward(relu_activation.a)
    loss = loss_activation.forward(output_layer.z, y)

    pred = np.argmax(loss_activation.output, axis = 1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis = 1)
    acc = np.mean(pred == y)

    if epoch % 100 == 0:
        print(f"acc = {acc} loss = {loss}")
        loss_list.append(loss)
        accuracy_list.append(acc)

    #backward pass
    loss_activation.backward(loss_activation.output, y)
    output_layer.backward(loss_activation.dinputs)
    relu_activation.backward(output_layer.dL_dx)
    hidden_layer_1.backward(relu_activation.dL_dz)

    #optimize
    optimizer.pre_update()
    optimizer.update_parameters(hidden_layer_1)
    optimizer.update_parameters(output_layer)
    optimizer.post_update()

plt.plot(np.arange(len(loss_list)),loss_list, color = 'b')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(True)
# plt.legend()
plt.show()

plt.plot(np.arange(len(accuracy_list)),accuracy_list, color = 'b')
plt.xlabel("epoch")
plt.ylabel("acc")
plt.grid(True)
# plt.legend()
plt.show()

np.savez('model_weights.npz', hidden_layer_1_w=hidden_layer_1.w, hidden_layer_1_b=hidden_layer_1.b,
         output_layer_w=output_layer.w, output_layer_b=output_layer.b)


