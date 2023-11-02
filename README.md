# Лабораторные работы для 2-го курса ФПЛ (2023/2024)

В рамках предмета
["Программирование для лингвистов"](https://www.hse.ru/edu/courses/835199210)
в НИУ ВШЭ - Нижний Новгород.

**Преподаватели:**

* [Демидовский Александр Владимирович](https://www.hse.ru/staff/demidovs) - лектор
* [Тугарёв Артём Михайлович](https://www.hse.ru/org/persons/224103384) - преподаватель практики
* [Казюлина Марина Сергеевна](https://github.com/marina-kaz) - преподаватель практики
* [Кащихин Андрей Николаевич](https://github.com/WhiteJaeger) - приглашённый эксперт
* [Жариков Егор Игоревич](https://t.me/godb0i) - ассистент
* [Новикова Ирина Алексеевна](https://t.me/iriinnnaaaaa) - ассистент
* [Блюдова Василиса Михайловна](https://t.me/Vasilisa282) - ассистент
* [Зайцева Вита Вячеславовна](https://t.me/v_ttec) - ассистент

**План лабораторных работ:**

1. [Определение языка текста на основе частотного словаря](./lab_1_classify_by_unigrams/README.md)
   1. Дедлайн: 6 октября
2. [Кодирование текста с помощью алгоритма `BPE`](./lab_2_tokenize_by_bpe/README.md)
   1. Дедлайн: 3 ноября
3. Лабораторная работа №3. TBD
   1. Дедлайн: XX ноября
4. Лабораторная работа №4. TBD
   1. Дедлайн: XX декабря

## История занятий

| Дата       | Тема лекции                                     | Тема практики. Материалы практики           |
|:-----------|:------------------------------------------------|:--------------------------------------------|
| 08.09.2023 | Установочная встреча. Историческая справка.     | Создание форка.                             |
| 15.09.2023 | Примитивные типы. Условия.                      | Настройка локальной машины.                 |
| 22.09.2023 | Строки: неизменяемые последовательности.        | Числа, условия, циклы, строки. [Листинг][1] |
| 29.09.2023 | Списки и кортежи.                               | Списки. [Листинг][2]                        |
| 06.10.2023 | Словари.                                        | Сдача лабораторной работы №1.               |
| 13.10.2023 | Функции.                                        | Словари. [Листинг][3] Функции. [Листинг][4] |
| 20.10.2023 | Введение в ООП. Класс как пользовательский тип. | Классы. [Листинг][5]                        |

Более полное содержание пройденных занятий в виде 
[списка ключевых тем](./docs/public/lectures_content_ru.md).

## Литература

### Базовый уровень

1. :books: :us: M. Lutz.
   [Learning Python](https://www.amazon.com/Learning-Python-5th-Mark-Lutz/dp/1449355730).
2. :video_camera: :ru: Хирьянов Т.Ф. Видеолекции.
   [Практика программирования на Python 3](https://www.youtube.com/watch?v=fgf57Sa5A-A&list=PLRDzFCPr95fLuusPXwvOPgXzBL3ZTzybY)
   . 2019.
3. :video_camera: :ru: Хирьянов Т.Ф. Видеолекции.
   [Алгоритмы и структуры данных на Python 3](https://www.youtube.com/watch?v=KdZ4HF1SrFs&list=PLRDzFCPr95fK7tr47883DFUbm4GeOjjc0)
   . 2017.
4. :bookmark: :us: [Официальная документация](https://docs.python.org/3/).

### Продвинутый уровень

1. :books: :us: M. Lutz.
   [Programming Python: Powerful Object-Oriented Programming](https://www.amazon.com/Programming-Python-Powerful-Object-Oriented/dp/0596158106)
2. :books: :us: J. Burton Browning.
   [Pro Python 3: Features and Tools for Professional Development](https://www.amazon.com/Pro-Python-Features-Professional-Development/dp/1484243846)
   . 
3. :video_camera: :ru: Хирьянов Т.Ф. Видеолекции.
   [Основы программирования и анализа данных на Python](https://teach-in.ru/course/python-programming-and-data-analysis-basics)
   . 2022.

## Порядок сдачи и оценивания лабораторной работы

1. Лабораторная работа допускается к очной сдаче.
2. Студент объяснил работу программы и показал её в действии.
3. Студент выполнил задание ментора по некоторой модификации кода.
4. Студент получает оценку:
    1. соответствующую ожидаемой, если все шаги выше выполнены и ментор удовлетворён ответом студента.
    2. на балл выше ожидаемой, если все шаги выше выполнены и ментор решает поощрить студента за отличный ответ.
    3. на балл ниже ожидаемой, если лабораторная работа сдана на неделю позже срока сдачи и выполнены критерии в 4.1.
    4. на два балла ниже ожидаемой, если лабораторная работа сдана на две недели и позже от срока сдачи и выполнены
       критерии в 4.1.

> **Замечание**: Студент может улучшить оценку по лабораторной работе,
> если после основной сдачи выполнит задания следующего уровня сложности
> относительно того уровня, на котором выполнялась реализация.

Лабораторная работа допускается к очной сдаче, если она:

1. представлена в виде пулл реквеста (Pull Request, PR) с правильно составленным названием по шаблону:
   `Laboratory work #<NUMBER>, <SURNAME> <NAME> - <UNIVERSITY GROUP NAME>`.
   1. Пример: `Laboratory work #1, Kashchikhin Andrey - 21FPL1`.
2. имеет заполненный файл `target_score.txt` с ожидаемой оценкой. Допустимые значения: 4, 6, 8, 10.
3. имеет "зелёный" статус - автоматические проверки качества и стиля кода, соответствующие заданной ожидаемой оценке,
   удовлетворены.
4. имеет лейбл `done`, выставленный ментором. Означает, что ментор посмотрел код студента и удовлетворён качеством кода.

## Ресурсы

1. [Таблица успеваемости](https://docs.google.com/spreadsheets/d/1mx9N7tmkaWjwK0h4oNnKFspjTheNVoDd)
2. [Инструкция по запуску unit тестов](./docs/public/tests.md)
3. [Инструкция по подготовке к прохождению курса](./docs/public/starting_guide_ru.md)
4. [Часто задаваемые вопросы](./docs/public/FAQ.md)

[1]: ./seminars/practice_2_string.py
[2]: ./seminars/practice_3_lists.py
[3]: ./seminars/practice_4_dicts.py
[4]: ./seminars/practice_5_functions.py
[5]: ./seminars/practice_6_classes.py
