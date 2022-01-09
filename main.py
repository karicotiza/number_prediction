import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras


class Net():
    def __init__(self, count):
        self.count = count
        self.sequence = []
        self.fill()
        self.model = tf.keras.Sequential(
            [keras.layers.Dense(units=1, input_shape=[1])]
        )
        self.model.compile(
            optimizer='adam',
            loss='mean_absolute_error'
        )

    def fill(self):
        print(f"Введите чисел - {self.count} ")
        for i in range(self.count):
            user_input = float(input(f"Число {i + 1}: "))
            self.sequence.append(user_input)

    def calculate(self):
        x = self.sequence.copy()
        x.pop(-1)
        x = np.array(x, dtype=float)

        y = self.sequence.copy()
        y.pop(0)
        y = np.array(y, dtype=float)

        self.model.add(tf.keras.layers.Dense(1))

        self.model.fit(x, y, epochs=10000)
        result = self.model.predict([self.sequence[-1]])

        return result


def main():
    net = Net(5)
    result = net.calculate()
    result = result[0][0]
    print(f"\nПредположительное следующее число - {result}")


if __name__ == "__main__":
    main()
