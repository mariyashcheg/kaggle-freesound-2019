# Kaggle Freesound 2019 (audio tagging)

## Архитектура
В бейзлайне использовалась сеть ResNet на log-mel спектрограммах. Хотелось более явно учитывать тот факт, что звук - это временной ряд, последовательность событий, поэтому заменяем на сеть, в которой есть рекуррентные блоки.

В ходе проекта были реализованы две архитектуры из следующих статей группы авторов, занимавших 1-2 места на Detection and
Classification of Acoustic Scenes and Events (DCASE2016 и DCASE2017) и последовательно модифицировавших свою сеть:
* "Convolutional Gated Recurrent Neural Network Incorporating Spatial Features for Audio Tagging" [https://arxiv.org/pdf/1702.07787.pdf](https://arxiv.org/pdf/1702.07787.pdf)
* "Attention and Localization based on a Deep Convolutional Recurrent Model for Weakly Supervised Audio Tagging" [https://arxiv.org/pdf/1703.06052.pdf](https://arxiv.org/pdf/1703.06052.pdf)
* "Large-Scale Weakly Supervised Audio Classification Using Gated Convolutional Neural Network" [https://arxiv.org/pdf/1710.00343.pdf](https://arxiv.org/pdf/1710.00343.pdf)

## Логи проекта
Первым шагом была реализована как самая современная и вобравшая в себя все лучшее сеть из последней статьи. Сеть состоит из:
* 4 сверточных блока: [Conv + BN + GLU]x2 + MP
* сверточный слой: Conv + MP
* двунаправленная однослойная GRU
* классифицирующая голова c локализацией: классификация FC + Sigmoid, локализация FC + Softmax  

На вход подаются log-mel спектрограммы (n_fft=5296, hop=2648, n_mels=64).  

![big_cgrnn](https://github.com/mariyashcheg/kaggle-freesound-2019/blob/master/img/big_cgrnn.png)  

Но сеть показывает плохое качество (lrap 0.06-0.07). 

Предположение: сеть сваливается в плохой локальный минимум и не может из него выбраться. Скорее всего шаг обучения из бейзлайна не подходит для данной сети. Подбор подходящего LR и использование циклического расписания не дают улучшения.
Графики нормы градиентов показывают, что на первой сотне итераций сеть учится, нормы градиентов большие, а потом что-то ломается и градиенты перестают течь. 

![gn_hist](https://github.com/mariyashcheg/kaggle-freesound-2019/blob/master/img/gn_hist.png)  

График распределения метрики _lwlrap_ по классам показывает, что сеть хорошо тегирует только один класс.  

Кажется, из-за большого количества блоков в архитектуре что-то могло пойти не так, поэтому было решено взять более простую архитектуру из первой статьи. Сеть:
* Conv + MP + ReLU
* двунаправленная трехслойная GRU
* FC + Sigmoid с усреднением по всем временным отрезкам

![simple_cgrnn](https://github.com/mariyashcheg/kaggle-freesound-2019/blob/master/img/simple_cgrnn.png)

Так как сеть более простая, на вход подаются спектрограммы больших размерностей (n_fft=1764, hop=220, n_mels=256). Но ситуация с плохим качеством, нулевыми градиентами и хорошим тегированием только одного класса повторяется.

Тогда добавляем в сеть блоки внимания и локализации из второй статьи. Работает!  

Отличие блока внимания+локализации в текущей архитектуре от блока локализации в первой архитектуре в том, что в текущей архитектуре они смотрят на входной тензор, а в первой архитектуре - на выходы рекуррентного слоя. 

![att](https://github.com/mariyashcheg/kaggle-freesound-2019/blob/master/img/attention.png)

Применяем этот подход к первой архитектуре - и она начинает обучаться!

Причем в результате первая архитектура быстрее сходится и достигает более высокого качества за то же количество эпох (вероятно засчет более тяжелого экстрактора фичей для рекуррентного слоя). Плюс обучение одной эпохи занимает в 2.5 раза меньше времени (первой сети на вход приходят тензоры меньшего размера).

![loss](https://github.com/mariyashcheg/kaggle-freesound-2019/blob/master/img/loss_gradnorm.png)  
![lrap](https://github.com/mariyashcheg/kaggle-freesound-2019/blob/master/img/lrap_lwlrap.png)  

