# Coursework - NeuroGena
## Репозиторий с файлами курсовой работы Герасименко Антона
Идея состоит в том чтоб генерировать файлы формата .xes - журналы событий с помощью нейронных сетей.
### В данном репозитории представлены 3 основных файла:
- # gen
Финальная версия генератора

Использование: `python gen.py <input>.xes <output>.xes <max_num_of_tokens>`

Где <input> и <output> название входного и выходного файла соответственно , а <max_num_of_tokens> - максимальное число сгенерированных новых токенов.
 
- # openAI
Альтернативная версия генератора с использованием технологий OpenAI.
  
**В файле нет токена OpenAI API без которого он не заработает**

Использование: `python gen.py <input>.xes <output>.xes <max_num_of_tokens>`

Где <input> и <output> название входного и выходного файла соответственно , а <max_num_of_tokens> - максимальное число сгенерированных новых токенов.
  
- # gpt2_train
Ноутбук в котором показан код, которым я обучал нейросеть gpt2 для генерации журналов событий.
Также в нем представлены некоторые эксперименты с другими сетями.
