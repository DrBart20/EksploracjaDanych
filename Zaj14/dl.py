import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Metraż (m2)
X = np.array([25, 30, 45, 50, 65, 80, 95, 110]).reshape(-1, 1)
# Cena (tys. PLN)
y = np.array([180, 210, 310, 350, 430, 520, 610, 700])

model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0001), loss='mse')

history = model.fit(X,y, epochs=200, verbose=0)

model.summary()

#Model ma 2 parametry: 1 wagę i 1 bias
#Waga reprezentuje cene za 1m2 i mowi o ile k złotych wzrośnie cena gdy metraz sie zwieskzy
#Bias to cena bazowa ktora mowi ile kosztowaloby mieszkanie o metrazu 0m2

X_new = np.array([[70]])
predicted_price = model.predict(X_new, verbose=0)

print(f"Przewidywana cena dla 70m2: {predicted_price[0][0]:.2f}k PLN")

learning_rates = [0.000001, 0.00001, 0.0001]
histories = {}

for lr in learning_rates:
    model_lr = keras.Sequential({
        keras.layers.Dense(units=1, input_shape=[1])
    })

    model_lr.compile(optimizer=keras.optimizers.SGD(learning_rate=lr), loss='mse')

    h = model_lr.fit(X,y, epochs=200, verbose=0)
    histories[lr] = h.history['loss']


plt.figure(figsize=(10,6))

colors = {0.000001:'red', 0.00001: 'blue', 0.0001: 'green'}

for lr, loss_history in histories.items():
    plt.plot(loss_history, label=f'LR = {lr}', color=colors[lr], linewidth=2)

plt.title('Wpływ Learning Rate na proces uczenia')
plt.xlabel('Numer epoki')
plt.ylabel('Błąd MSE')
plt.grid(True)
plt.legend()
plt.show()