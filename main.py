import matplotlib.pyplot as plt

def model_analyze(log):
  # model = keras.model.load_model(path)
  # Выводим график точности на обучающей выборке
  # label - имя графика в легенде
  plt.plot(log.history['accuracy'], 
          label='Доля верных ответов на обучающем наборе')
  # Выводим график точности на проверочной выборке
  plt.plot(log.history['val_accuracy'], 
          label='Доля верных ответов на проверочном наборе')

  # Выводим подписи осей
  plt.title("Точность")
  plt.xlabel('Эпоха обучения')
  plt.ylabel('Доля верных ответов')

  # Выводим легенду
  plt.legend()
  plt.show()

  plt.plot(log.history['loss'], label='Ошибка на обучающем наборе')
  plt.plot(log.history['val_loss'], label='Ошибка на проверочном наборе')
  plt.title("Ошибка")
  plt.xlabel('Эпоха обучения')
  plt.ylabel('Ошибка')
  plt.legend()
  plt.show()
