import numpy as np


class SoftmaxLogReg:

    def __init__(self):
        self.loss = []  # значения функции потерь
        self.thetas = None  # веса для каждого из классов

    def fit(self, x, y, iterator=30000, learning_rate=0.002):
        # добавляем вектор единиц, чтобы потом в скалярном произведении с весами получить вектор свообдных коэффициентов
        self.add_ones(x)
        # кодируем таргет переменные
        y_ohe = self.one_hot_encoding(y)
        # изначально все веса равны нулю
        self.thetas = np.zeros((x.shape[1], y_ohe.shape[1]))

        # минимизируем функцию ошибок 30000 раз, подгоняя веса модели
        for i in range(iterator):
            # получаем вероятности, минимизируем веса, идя против вектора градиента
            probabilites = self.stable_softmax(x)
            grad = self.gradient_softmax(x, probabilites, y_ohe)
            self.thetas -= learning_rate * grad

            # запоминаем промежуточные значения функции потерь
            if i % 1000 == 0:
                self.loss.append(self.cross_entropy(probabilites, y_ohe, epsilon=1e-9))

    # предикт модели - максимум из софтмаксов
    def predict(self, x):
        # добавим столбец свободных коэффициентов
        self.add_ones(x)
        return np.argmax(self.stable_softmax(x), axis=1)

    # функция softmax
    def stable_softmax(self, x):
        # создаем логиты
        z = np.dot(-x, self.thetas)
        # нормируем логиты, чтобы экспоненты не улетали в бесконечность
        z = z - np.max(z, axis=-1, keepdims=True)
        # создаем числитель, затем знаменатель и вычисляем функцию
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        softmax = numerator / denominator
        return softmax

    # функция потерь
    @staticmethod
    def cross_entropy(probabilites, y_ohe, epsilon=1e-9):
        # для нормирования функции поделим её на количество экземпляров
        n = probabilites.shape[0]
        # константа в логарифме для вычислительной устойчивости
        ce = -np.sum(y_ohe * np.log(probabilites + epsilon)) / n
        return ce

    # считаем градиент функции потерь
    @staticmethod
    def gradient_softmax(x, probabilites, y_ohe):
        return np.array(1 / probabilites.shape[0] * np.dot(x.T, (y_ohe - probabilites)))

    # добавляем столбец свободных коэффициентов
    @staticmethod
    def add_ones(x):
        return x.insert(0, 'x0', np.ones(x.shape[0]))

    # создаем закодированные таргет переменные
    @staticmethod
    def one_hot_encoding(y):
        # количество примеров и количество классов
        examples, features = y.shape[0], len(np.unique(y))
        # нулевая матрица: количество наблюдений x количество признаков
        zeros_matrix = np.zeros((examples, features))
        # построчно проходимся по нулевой матрице и с помощью индекса заполняем соответствующее значение единицей
        for i, (row, digit) in enumerate(zip(zeros_matrix, y)):
            zeros_matrix[i][digit] = 1

        return zeros_matrix
