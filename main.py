import matplotlib.pyplot as plt
from time import gmtime, strftime


def model_analyze(log):
  print('Final accuracy on test batch is ', round(log.history['val_accuracy'][-1]*100, 2), '%', sep = '')
  descr = input('Enter description: ')
  if (descr == ''): descr = 'Plot'
  
  fig, axis = plt.subplots(1, 2, figsize = (16, 6))

  axis[0].plot(log.history['accuracy'], 
          label='Доля верных ответов на обучающем наборе')
  # Выводим график точности на проверочной выборке
  axis[0].plot(log.history['val_accuracy'], 
          label='Доля верных ответов на проверочном наборе')
  

  # Выводим подписи осей
  axis[0].set_title('Точность')
  axis[0].set_xlabel('Эпоха обучения')
  axis[0].set_ylabel('Доля верных ответов')

  axis[0].set_yticks([i/100 for i in range(90, 101, 1)])
  axis[0].legend()

  axis[1].plot(log.history['loss'], label='Ошибка на обучающем наборе')
  axis[1].plot(log.history['val_loss'], label='Ошибка на проверочном наборе')
  axis[1].set_title("Ошибка")
  axis[1].set_xlabel('Эпоха обучения')
  axis[1].set_ylabel('Ошибка')
  axis[1].legend()

  fig.suptitle(descr)
  plt.savefig(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '__' + descr + '.png')
  plt.show()
