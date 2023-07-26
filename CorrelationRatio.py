import numpy as np


# представляет из себя отношение суммы внутригрупповых дисперсий на общую дисперсию
def correlation_ratio(numerical, categorical):
    # рассчитаем общую дисперсию
    values = np.array(numerical)
    ss_total = np.sum((values.mean() - values) ** 2)

    # закодировали таргет переменные числами
    cats = np.unique(categorical, return_inverse=True)[1]

    # объявим переменную для внутригрупповой дисперсии
    ss_ingroups = 0

    # в цикле, состоящем из количества категорий
    for c in np.unique(cats):
        # вычленим группу оценок по каждому предмету
        group = values[np.argwhere(cats == c).flatten()]
        # найдем суммы квадратов отклонений значений от групповых средних
        # и сложим эти результаты для каждой группы
        ss_ingroups += np.sum((group.mean() - group) ** 2)

    # вычитаем из единицы и берём корень
    return np.sqrt(1 - ss_ingroups / ss_total)
