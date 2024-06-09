# Лекція 6.2. Загальні пояття колекції та складних структур даних

#Рядки
Python дозволяє вставляти в рядок значення інших змінних. Для цього всередині рядка змінні розміщуються у фігурних дужках {}, а перед усім рядком ставиться символ f:


```python
userName = "Андрій"
userAge = 37
user = f"Ім'я: {userName} та вік: {userAge}"
print(user)
```

    Ім'я: Андрій та вік: 37
    

Також ми можемо звернутися до окремих символів рядка за індексом у квадратних дужках:


```python
string = "hello world"
c0 = string[0]  # h
print(c0)
c6 = string[6]  # w
print(c6)

c11 = string[11]  # помилка IndexError: string index out of range
print(c11)
```

    h
    w
    


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-2-2b3fb53fd5f1> in <cell line: 7>()
          5 print(c6)
          6 
    ----> 7 c11 = string[11]  # помилка IndexError: string index out of range
          8 print(c11)
    

    IndexError: string index out of range


За допомогою циклу for можна перебрати всі символи рядка:


```python
string = "hello world"
for char in string:
    print(char, end = " ")
```

    h e l l o   w o r l d 

За потреби ми можемо отримати з рядка не тільки окремі символи, а й підрядок.


```python
string = "hello world"

sub_string1 = string[:5]
print(sub_string1)      # hello

sub_string2 = string[2:5]
print(sub_string2)      # llo

sub_string3 = string[2:9:2]
print(sub_string3)      # lowr
```

    hello
    llo
    lowr
    

Однією з найпоширеніших операцій з рядками є їх об'єднання або конкатенація. Для об'єднання рядків застосовується операція додавання:


```python
name = "Andrii"
surname = "Nikitenko"
fullname = name + " " + surname
print(fullname)
```

    Andrii Nikitenko
    

Для повторення рядка певну кількість разів застосовується операція множення:


```python
print("a" * 3)  # aaa
print("he" * 4)  # hehehehe
```

    aaa
    hehehehe
    

Особливо слід звернути увагу на порівняння рядків. Під час порівняння рядків беруть до уваги символи та їхній регістр. Так, цифровий символ умовно менший, ніж будь-який алфавітний символ. Алфавітний символ у верхньому регістрі умовно менший, ніж алфавітні символи в нижньому регістрі. Наприклад:


```python
str1 = "1a"
str2 = "aa"
str3 = "Aa"
print(str1 > str2)  # False, оскільки перший символ у str1 - цифра
print(str2 > str3)  # True, оскільки перший символ у str2 - у нижньому регістрі
```

    False
    True
    

Залежність від регістра не завжди бажана, оскільки по суті ми маємо справу з однаковими рядками. У цьому випадку перед порівнянням ми можемо привести обидва рядки до одного з регістрів.

Функція lower() приводить рядок до нижнього регістру, а функція upper() - до верхнього.


```python
str1 = "Tom"
str2 = "tom"
print(str1 == str2)  # False - рядки не рівні

print(str1.lower() == str2.lower())  # True
```

    False
    True
    

Для отримання довжини рядка можна використовувати функцію len():


```python
string = "hello world"
length = len(string)
print(length)   # 11
```

    11
    

За допомогою виразу **term in string** можна знайти підрядок term у рядку string. Якщо підрядок знайдено, то вираз поверне значення True, інакше повертається значення False:


```python
string = "hello world"
exist = "hello" in string
print(exist)    # True

exist = "sword" in string
print(exist)    # False
```

    True
    False
    

Тепер розглянемо основні методи:


```python
file_name = "hello.py"

starts_with_hello = file_name.startswith("hello")
ends_with_exe = file_name.endswith("exe")
print("Starts with: {0}, ends with {1}".format(starts_with_hello, ends_with_exe))
```

    Starts with: True, ends with False
    

Видалення пробілів на початку та в кінці рядка:


```python
string = "   hello  world!  "
string = string.strip()
print(string)           # hello  world!
```

    hello  world!
    

Для заміни в рядку одного підрядка на інший застосовується метод replace():


```python
phone = "+1-234-567-89-10"

# заміна дефісів на пробіл
edited_phone = phone.replace("-", " ")
print(edited_phone) # +1 234 567 89 10

# видалення дефісів
edited_phone = phone.replace("-", "")
print(edited_phone) # +12345678910

# заміна тільки першого дефіса
edited_phone = phone.replace("-", "", 1)
print(edited_phone) # +1234-567-89-10
```

    +1 234 567 89 10
    +12345678910
    +1234-567-89-10
    

Метод split() розбиває рядок на список підрядків залежно від роздільника. Як роздільник може виступати будь-який символ або послідовність символів. Цей метод має такі форми:


```python
text = "Це був величезний, у два обхвати дуб, з обламаними гілками і з обламаною корою"
# поділ за пробілами
splitted_text = text.split()
print(splitted_text)
print(splitted_text[6])     # дуб,

# розбиття за комами
splitted_text = text.split(",")
print(splitted_text)
print(splitted_text[1])     # у два обхвати дуб

# розбиття за першими п'ятьма пробілами
splitted_text = text.split(" ", 5)
print(splitted_text)
print(splitted_text[5])     # обхвати дуб, з обламаними гілками і з обламаною корою
```

    ['Це', 'був', 'величезний,', 'у', 'два', 'обхвати', 'дуб,', 'з', 'обламаними', 'гілками', 'і', 'з', 'обламаною', 'корою']
    дуб,
    ['Це був величезний', ' у два обхвати дуб', ' з обламаними гілками і з обламаною корою']
     у два обхвати дуб
    ['Це', 'був', 'величезний,', 'у', 'два', 'обхвати дуб, з обламаними гілками і з обламаною корою']
    обхвати дуб, з обламаними гілками і з обламаною корою
    

Під час розгляду найпростіших операцій з рядками було показано, як об'єднувати рядки за допомогою операції додавання. Іншу можливість для з'єднання рядків представляє метод join(): він об'єднує список рядків. Причому поточний рядок, у якого викликається цей метод, використовується як роздільник:


```python
words = ["Слава", "Україні!", "Героям", "Слава!"]

# роздільник - пробіл
sentence = " ".join(words)
print(sentence)  # Слава Україні! Героям Слава!

# роздільник - вертикальна риска
sentence = " | ".join(words)
print(sentence)  # Слава | Україні! | Героям | Слава!
```

    Слава Україні! Героям Слава!
    Слава | Україні! | Героям | Слава!
    

# Форматування рядків

У минулих темах було розглянуто, як можна вставляти в рядок деякі значення, випереджаючи рядок символом f:


```python
first_name="Andrii"
text = f"Добрий день, {first_name}."
print(text)     # Добрий день, Andrii.

name="Dmytro"
age=23
info = f"Ім'я: {name}\t Вік: {age}"
print(info)     # Ім'я: Dmytro	 Вік: 23
```

    Добрий день, Andrii.
    Ім'я: Dmytro	 Вік: 23
    

Але також у Python є альтернативний спосіб, який надає метод format(). Цей метод дає змогу форматувати рядок, вставляючи в нього на місце плейсхолдерів певні значення.

У рядку, що форматується, ми можемо визначати параметри, у методі format() передавати для цих параметрів значення:


```python
text = "Добрий день, {first_name}.".format(first_name="Andrii")
print(text)     # Добрий день, Andrii.

info = "Ім'я: {name}\t Вік: {age}".format(name="Дмитро", age=23)
print(info)     # Ім'я: Дмитро	 Вік: 23
```

    Добрий день, Andrii.
    Ім'я: Дмитро	 Вік: 23
    

Ми також можемо послідовно передавати в метод format набір аргументів, а в самому рядку, що форматується, вставляти ці аргументи, вказуючи у фігурних дужках їхній номер (нумерація починається з нуля):


```python
info = "Ім'я: {0}\t Вік: {1}".format("Андрій", 23)
print(info)     # Name: Bob  Age: 23

text = "Добрий день, {0} {0} {0}.".format("Дмитро")
```

    Ім'я: Андрій	 Вік: 23
    

Ще один спосіб передавання формованих значень у рядок - використання підстановок або спеціальних плейсхолдерів, на місце яких вставляються певні значення. Для форматування ми можемо використовувати такі плейсхолдери:


1. s: для вставки рядків
2. d: для вставки цілих чисел
3. f: для вставки дробових чисел. Для цього типу також можна визначити через крапку кількість знаків у дробовій частині.
4. %: множить значення на 100 і додає знак відсотка
5. e: виводить число в експоненціальному записі

Під час виклику методу format у нього як аргументи передаються значення, які вставляються на місце плейсхолдерів:






```python
welcome = "Добрий день, {:s}"
name = "Дмитро"
formatted_welcome = welcome.format(name)
print(formatted_welcome)        # Добрий день, Дмитро
```

    Добрий день, Дмитро
    

Форматування цілих чисел:


```python
source = "{:d} символів"
number = 5
target = source.format(number)
print(target)   # 5 символів
```

    5 символів
    

Для дробових чисел, тобто таких, які представляють тип float, перед кодом плейсхолдера після крапки можна вказати, скільки знаків у дробовій частині ми хочемо вивести:


```python
number = 23.8589578
print("{:.2f}".format(number))   # 23.86
print("{:.3f}".format(number))   # 23.859
print("{:.4f}".format(number))   # 23.8590
print("{:,.2f}".format(10001.23554))    # 10,001.24
```

    23.86
    23.859
    23.8590
    10,001.24
    

Для виведення відсотків краще скористатися кодом "%":


```python
number = .12345
print("{:%}".format(number))        # 12.345000%
print("{:.0%}".format(number))      # 12%
print("{:.1%}".format(number))      # 12.3%

print(f"{number:%}")        # 12.345000%
print(f"{number:.0%}")      # 12%
print(f"{number:.1%}")      # 12.3%
```

    12.345000%
    12%
    12.3%
    12.345000%
    12%
    12.3%
    

Для виведення числа в експоненціальному записі використовується плейсхолдер "e":


```python
number = 12345.6789
print("{:e}".format(number))        # 1.234568e+04
print("{:.0e}".format(number))      # 1e+04
print("{:.1e}".format(number))      # 1.2e+04

print(f"{number:e}")        # 1.234568e+04
print(f"{number:.0e}")      # 1e+04
print(f"{number:.1e}")      # 1.2e+04
```

    1.234568e+04
    1e+04
    1.2e+04
    1.234568e+04
    1e+04
    1.2e+04
    

# List (списки)
Для створення списку застосовуються квадратні дужки [], усередині яких через кому перераховуються елементи списку. Наприклад, визначимо список чисел:


```python
numbers = [1, 2, 3, 4, 5]
```

Подібним чином можна визначати списки з даними інших типів, наприклад, визначимо список рядків:


```python
people = ["Tom", "Sam", "Bob"]
```

Є два варіанти створення списків - вони ідентичні


```python
numbers1 = []
numbers2 = list()
```

# Додатково!
Існує ще один спосіб створення списків *List comprehension*, дана функціональність надає більш короткий і лаконічний синтаксис для створення списків на основі інших наборів даних.
### newlist = [expression for item in iterable (if condition)]

Припустимо, нам треба вибрати зі списку всі числа, які більші за 0


```python
numbers = [-3, -2, -1, 0, 1, 2, 3]
positive_numbers = []
for n in numbers:
    if n > 0:
        positive_numbers.append(n)

print(positive_numbers)     # [1, 2, 3]
```

    [1, 2, 3]
    

Тепер зробимо цю задачу за допомогою list comprehension


```python
numbers = [-3, -2, -1, 0, 1, 2, 3]
positive_numbers = [n for n in numbers if n > 0]

print(positive_numbers)     # [1, 2, 3]
```

    [1, 2, 3]
    


```python
numbers = [-3, -2, -1, 0, 1, 2, 3]
new_numbers = [n * 2 for n in numbers]
print(new_numbers)      # [-6, -4, -2, 0, 2, 4, 6]
```

    [-6, -4, -2, 0, 2, 4, 6]
    

Список необов'язково повинен містити тільки однотипні об'єкти.


```python
objects = [1, 2.6, "Hello", True]
```

Для перевірки елементів списку можна використовувати стандартну функцію print, яка виводить вміст списку в читабельному вигляді:


```python
numbers = [1, 2, 3, 4]
people = ["Андрій", "Дмитро", "Марк"]

print(numbers)  # [1, 2, 3, 4]
print(people)   # ['Андрій', 'Дмитро', 'Марк']
```

    [1, 2, 3, 4]
    ['Андрій', 'Дмитро', 'Марк']
    

Конструктор list може приймати набір значень, на основі яких створюється список:


```python
numbers1 = [1, 2, 3, 4]
numbers2 = list(numbers1)
print(numbers2)  # [1, 2, 3, 4]

letters = list("Привіт")
print(letters)      # ['П', 'р', 'и', 'в', 'і', 'т']
```

    [1, 2, 3, 4]
    ['П', 'р', 'и', 'в', 'і', 'т']
    

Якщо необхідно створити список, у якому повторюється одне й те саме значення кілька разів, то можна використати символ зірочки *, тобто фактично застосувати операцію множення до вже наявного списку:


```python
numbers = [5] * 6
print(numbers)

people = ["Андрій"] * 3
print(people)

students = ["Дмитро", "Марк"] * 2
print(students)
```

    [5, 5, 5, 5, 5, 5]
    ['Андрій', 'Андрій', 'Андрій']
    ['Дмитро', 'Марк', 'Дмитро', 'Марк']
    

Для звернення до елементів списку треба використовувати індекси, які представляють номер елемента у списку. Індекси починаються з нуля. Тобто перший елемент матиме індекс 0, другий елемент - індекс 1 і так далі. Для звернення до елементів з кінця можна використовувати від'ємні індекси, починаючи з -1. Тобто останній елемент матиме індекс -1, передостанній - -2 і так далі.


```python
people = ["Андрій", "Дмитро", "Марк"]
print(people[0])
print(people[1])
print(people[2])


print(people[-2])
print(people[-1])
print(people[-3])
```

    Андрій
    Дмитро
    Марк
    Дмитро
    Марк
    Андрій
    

Для зміни елемента списку достатньо присвоїти йому нове значення:


```python
people = ["Андрій", "Дмитро", "Марк"]

people[1] = "Микола"
print(people[1])
print(people)
```

    Микола
    ['Андрій', 'Микола', 'Марк']
    

Розкладання списку
Python дозволяє розкласти список на окремі елементи:


```python
people = ["Андрій", "Дмитро", "Марк"]

andrii, dmytro, mark = people

print(andrii)
print(dmytro)
print(mark)
```

    Андрій
    Дмитро
    Марк
    

Для перебору елементів можна використовувати як цикл for, так і цикл while.
Перебір за допомогою циклу for:


```python
people = ["Андрій", "Дмитро", "Марк"]
for person in people:
    print(person)
```

    Андрій
    Дмитро
    Марк
    

Два списки вважаються рівними, якщо вони містять один і той самий набір елементів:


```python
numbers1 = [1, 2, 3, 4]
numbers2 = list([1, 2, 3, 4])
if numbers1 == numbers2:
    print("numbers1 equal to numbers2")
else:
    print("numbers1 is not equal to numbers2")
```

    numbers1 equal to numbers2
    

Якщо необхідно отримати якусь певну частину списку, то ми можемо застосовувати спеціальний синтаксис


```python
people = ["Андрій", "Дмитро", "Марк", "Тетяна", "Анастасія", "Анна"]

slice_people1 = people[:3]   # с 0 по 3
print(slice_people1)   # ['Андрій', 'Дмитро', 'Марк']

slice_people2 = people[1:3]   # с 1 по 3
print(slice_people2)   # ['Дмитро', 'Марк']

slice_people3 = people[1:6:2]   # с 1 по 6 з кроком 2
print(slice_people3)   # ['Дмитро', 'Тетяна', 'Анна']
```

    ['Андрій', 'Дмитро', 'Марк']
    ['Дмитро', 'Марк']
    ['Дмитро', 'Тетяна', 'Анна']
    

Для додавання елемента застосовуються методи append(), extend і insert, а для видалення - методи remove(), pop() і clear().


```python
people = ["Tom", "Bob"]

# додаємо в кінець списку
people.append("Alice")  # ["Tom", "Bob", "Alice"]
# додаємо на другу позицію
people.insert(1, "Bill")  # ["Tom", "Bill", "Bob", "Alice"]
# додаємо набір елементів ["Mike", "Sam"]
people.extend(["Mike", "Sam"])      # ["Tom", "Bill", "Bob", "Alice", "Mike", "Sam"]
# отримуємо індекс елемента
index_of_tom = people.index("Tom")
# видаляємо за цим індексом
removed_item = people.pop(index_of_tom)     # ["Bill", "Bob", "Alice", "Mike", "Sam"]
# видаляємо останній елемент
last_item = people.pop()     # ["Bill", "Bob", "Alice", "Mike"]
# видаляємо елемент "Alice"
people.remove("Alice")      # ["Bill", "Bob", "Mike"]
print(people)       # ["Bill", "Bob", "Mike"]
# видаляємо всі елементи
people.clear()
print(people)       # []
```

    ['Bill', 'Bob', 'Mike']
    []
    

Якщо певний елемент не знайдено, то методи remove та index генерують виняток. Щоб уникнути подібної ситуації, перед операцією з елементом можна перевіряти його наявність за допомогою ключового слова in:


```python
people = ["Tom", "Bob", "Alice", "Sam"]

if "Alice" in people:
    people.remove("Alice")
print(people)       # ["Tom", "Bob", "Sam"]
```

    ['Tom', 'Bob', 'Sam']
    

Python також підтримує ще один спосіб видалення елементів списку - за допомогою оператора del. Як параметр цьому оператору передається елемент, що видаляється, або набір елементів:


```python
people = ["Tom", "Bob", "Alice", "Sam", "Bill", "Kate", "Mike"]

del people[1]
print(people)
del people[:3]
print(people)
del people[1:]
print(people)
```

    ['Tom', 'Alice', 'Sam', 'Bill', 'Kate', 'Mike']
    ['Bill', 'Kate', 'Mike']
    ['Bill']
    

Якщо необхідно дізнатися, скільки разів у списку присутній той чи інший елемент, то можна застосувати метод count():


```python
people = ["Tom", "Bob", "Alice", "Tom", "Bill", "Tom"]

people_count = people.count("Tom")
print(people_count)      # 3
```

    3
    

Для сортування за зростанням застосовується метод sort():


```python
people = ["Tom", "Bob", "Alice", "Sam", "Bill"]

people.sort()
print(people)      # ["Alice", "Bill", "Bob", "Sam", "Tom"]
```

    ['Alice', 'Bill', 'Bob', 'Sam', 'Tom']
    

Якщо необхідно відсортувати дані у зворотному порядку, то ми можемо після сортування застосувати метод reverse():


```python
people = ["Tom", "Bob", "Alice", "Sam", "Bill"]

people.sort()
people.reverse()
print(people)      # ["Tom", "Sam", "Bob", "Bill", "Alice"]
```

    ['Tom', 'Sam', 'Bob', 'Bill', 'Alice']
    

Вбудовані функції Python min() і max() дають змогу знайти мінімальне і максимальне значення відповідно:


```python
numbers = [9, 21, 12, 1, 3, 15, 18]
print(min(numbers))     # 1
print(max(numbers))     # 21
```

    1
    21
    

Для об'єднання списків застосовується операція додавання (+):


```python
people1 = ["Tom", "Bob", "Alice"]
people2 = ["Tom", "Sam", "Tim", "Bill"]
people3 = people1 + people2
print(people3)   # ["Tom", "Bob", "Alice", "Tom", "Sam", "Tim", "Bill"]
```

    ['Tom', 'Bob', 'Alice', 'Tom', 'Sam', 'Tim', 'Bill']
    

Списки крім стандартних даних типу рядків, чисел, також можуть містити інші списки. Подібні списки можна асоціювати з таблицями, де вкладені списки виконують роль рядків. Наприклад:


```python
people = [["Андрій", 20], ["Дмитро", 31], ["Марк", 12]]

print(people[0])         # ['Андрій', 20]
print(people[0][0])      # Андрій
print(people[0][1])      # 20
```

    ['Андрій', 20]
    Андрій
    20
    


```python
people = [
    ["Андрій", 20],
    ["Дмитро", 31],
    ["Марк", 12]
]

for person in people:
    for item in person:
        print(item, end=" | ")
```

    Андрій | 20 | Дмитро | 31 | Марк | 12 | 


```python
people = ["Tom", "bob", "alice", "Sam", "Bill"]

people.sort()       # стандартне сортування
print(people)      # ["Bill", "Sam", "Tom", "alice", "bob"]

people.sort(key=str.lower)  # сортування без урахування регістра
print(people)      # ["alice", "Bill", "bob", "Sam", "Tom"]
```

    ['Bill', 'Sam', 'Tom', 'alice', 'bob']
    ['alice', 'Bill', 'bob', 'Sam', 'Tom']
    

# Кортежі
Кортеж (tuple) представляє послідовність елементів, яка багато в чому схожа на список за тим винятком, що кортеж є незмінним (immutable) типом. Тому ми не можемо додавати або видаляти елементи в кортежі, змінювати його.
Для створення кортежу використовуються круглі дужки, в які поміщаються його значення, розділені комами:


```python
person = ("Andrii", 30)
print(person) # ("Andrii", 30)
```

    ('Andrii', 30)
    

Також для визначення кортежу ми можемо просто перерахувати значення через кому без застосування дужок:


```python
person = "Andrii",
print(person) # ("Andrii", 30)
```

    ('Andrii',)
    

Якщо раптом кортеж складається з одного елемента, то після єдиного елемента кортежу необхідно поставити кому:


```python
#person = ("Andrii",)
person = ("Andrii")
print(type(person))
```

    <class 'str'>
    

Для створення кортежу з іншого набору елементів, наприклад, зі списку, можна передати список у функцію tuple(), яка поверне кортеж:


```python
data = ["Andrii", 30, "DonNTU"]
person = tuple(data)
print(person)      # ("Andrii", 30, "DonNTU")
```

    ('Andrii', 30, 'DonNTU')
    

За допомогою вбудованої функції len() можна отримати довжину кортежу:


```python
data = ["Andrii", 30, "DonNTU"]
print(len(data))     # 3
```

    3
    

Звернення до елементів у кортежі відбувається також, як і в списку, за індексом. Індексація починається також з нуля при отриманні елементів з початку списку і з -1 при отриманні елементів з кінця списку:


```python
person = ("Andrii", 30, "DonNTU", "software developer")
print(person[0]) # Andrii
print(person[1]) # 30
print(person[-1]) # software developer
```

    Andrii
    30
    software developer
    

Але оскільки кортеж - незмінний тип (immutable), то ми не зможемо змінити його елементи. Тобто наступний запис працювати не буде:


```python
person[1] = "Dmytro"
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-64-a31b864045fb> in <cell line: 1>()
    ----> 1 person[1] = "Dmytro"
    

    TypeError: 'tuple' object does not support item assignment


За необхідності ми можемо розкласти кортеж на окремі змінні:


```python
name, age, company, position = ("Andrii", 30, "DonNTU", "software developer")
print(name)         # Andrii
print(age)          # 30
print(position)     # software developer
print(company)     # DonNTU
```

    Andrii
    30
    software developer
    DonNTU
    

Як і в списках, можна отримати частину кортежа у вигляді іншого кортежа


```python
person = ("Andrii", 30, "DonNTU", "software developer")

# отримаємо підкортеж з 1 по 3 елементи (не включаючи)
print(person[1:3])     # (30, "DonNTU")

# отримаємо підкортеж з 0 по 3 елементи (не включаючи)
print(person[:3])     # ("Andrii", 30, "DonNTU")

# отримаємо підкортеж з 1 по останній елемент
print(person[1:])     # (30, "DonNTU", "software developer")
```

    (30, 'DonNTU')
    ('Andrii', 30, 'DonNTU')
    (30, 'DonNTU', 'software developer')
    

Особливо зручно використовувати кортежі, коли необхідно повернути з функції одразу кілька значень. Коли функція повертає кілька значень, фактично вона повертає в кортеж:


```python
def get_user():
    name = "Andrii"
    age = 30
    company = "DonNTU"
    return name, age, company


user = get_user()
print(user)     # ("Andrii", 30, "DonNTU")
print(type(user))
```

    ('Andrii', 30, 'DonNTU')
    <class 'tuple'>
    

При передачі кортежу у функцію за допомогою оператора * його можна розкласти на окремі значення, які передаються параметрам функції:


```python
def print_person(name, age, company):
    print(f"Ім'я: {name}  Вік: {age}  Компанія: {company}")

andrii = ("Andrii", 30)
print_person(*andrii, "DonNTU")     # Ім'я: Andrii  Вік: 30  Компанія: DonNTU

dmytro = ("Dmytro", 54, "Google")
print_person(*dmytro)      # Ім'я: Dmytro  Вік: 54  Компанія: Google
```

    Ім'я: Andrii  Вік: 30  Компанія: DonNTU
    Ім'я: Dmytro  Вік: 54  Компанія: Google
    

Для перебору кортежу можна використовувати стандартні цикли for і while. За допомогою циклу for:


```python
andrii = ("Andrii", 30, "DonNTU")
for item in andrii:
    print(item)

# за допомогою циклу while
dmytro = ("Dmytro", 54, "Google")

i = 0
while i < len(dmytro):
    print(dmytro[i])
    i += 1
```

    Andrii
    30
    DonNTU
    Dmytro
    54
    Google
    

Як для списку за допомогою виразу елемент in кортеж можна перевірити наявність елемента в кортежі:


```python
user = ("Andrii", 30, "DonNTU")
name = "Dmytro"
if name in user:
    print("Користувача звуть Андрій")
else:
    print("Користувач має інше ім'я")
```

    Користувач має інше ім'я
    

# Множина
Множина (set) представляє ще один вид набору, який зберігає тільки унікальні елементи. Для визначення множини використовуються фігурні дужки, в яких перераховуються елементи:


```python
users = {"Andrii", "Dmytro", "Eliza", "Dmytro"}
print(users)    # {'Dmytro', 'Eliza', 'Andrii'}
```

    {'Dmytro', 'Andrii', 'Eliza'}
    

Також для визначення множини може застосовуватися функція set(), в яку передається список або кортеж елементів:


```python
people = ["Andrii", "Dmytro", "Eliza"]
users = set(people)
print(users)    # {'Dmytro', 'Eliza', 'Andrii'}
```

    {'Dmytro', 'Andrii', 'Eliza'}
    

Функцію set зручно застосовувати для створення порожньої множини:


```python
users = set()
```

Для отримання довжини множини застосовується вбудована функція len():


```python
users = {"Andrii", "Dmytro", "Eliza"}
print(len(users))       # 3
```

    3
    

Для додавання одиночного елемента викликається метод add():


```python
users = set()
users.add("Sam")
print(users)
```

    {'Sam'}
    

Для видалення одного елемента викликається метод remove(), у який передається елемент, що видаляється. Але слід враховувати, що якщо такого елемента не виявиться в множині, то буде згенеровано помилку. Тому перед видаленням слід перевіряти на наявність елемента за допомогою оператора in:


```python
users = {"Andrii", "Dmytro", "Eliza"}

user = "Andrii"

users.remove(user)
users.remove("Bob")
print(users)    # {'Dmytro', 'Eliza'}
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-77-1bba8fbf678c> in <cell line: 6>()
          4 
          5 users.remove(user)
    ----> 6 users.remove("Bob")
          7 print(users)    # {'Dmytro', 'Eliza'}
    

    KeyError: 'Bob'


Також для видалення можна використовувати метод discard(), який не буде генерувати винятки за відсутності елемента:


```python
users = {"Andrii", "Dmytro", "Eliza"}

users.discard("Bob")    # елемент "Bob" відсутній, і метод нічого не робить
print(users)    #  {'Dmytro', 'Eliza', 'Andrii'}

users.discard("Andrii")    # елемент "Andrii" є, і метод видаляє елемент
print(users)    # {'Dmytro', 'Eliza'}
```

    {'Dmytro', 'Andrii', 'Eliza'}
    {'Dmytro', 'Eliza'}
    

Для видалення всіх елементів викликається метод clear():


```python
users.clear()
```

Для перебору елементів можна використовувати цикл for:


```python
users = {"Andrii", "Dmytro", "Eliza"}

for user in users:
    print(user)
```

    Dmytro
    Andrii
    Eliza
    

За допомогою методу copy() можна скопіювати вміст однієї множини в іншу змінну:


```python
users = {"Andrii", "Dmytro", "Eliza"}
students = users.copy()
print(students)     # {"Andrii", "Dmytro", "Eliza"}
```

    {'Dmytro', 'Andrii', 'Eliza'}
    

Метод union() об'єднує дві множини і повертає нову множину:


```python
users = {"Andrii", "Dmytro", "Eliza"}
users2 = {"Kate", "Tetiana", "Mark"}

users3 = users.union(users2)
print(users3)   # {'Eliza', 'Kate', 'Andrii', 'Mark', 'Dmytro', 'Tetiana'}
```

    {'Tetiana', 'Andrii', 'Mark', 'Eliza', 'Kate', 'Dmytro'}
    

Перетин множин дає змогу отримати тільки ті елементи, які є одночасно в обох множинах. Метод intersection() виконує операцію перетину множин і повертає нову множину:


```python
users = {"Andrii", "Dmytro", "Eliza"}
users2 = {"Eliza", "Dmytro", "Mark"}

users3 = users.intersection(users2)
print(users3)   # {'Dmytro', 'Eliza'}
```

    {'Dmytro', 'Eliza'}
    

Замість методу intersection ми могли б використовувати операцію логічного множення:


```python
users = {"Andrii", "Dmytro", "Eliza"}
users2 = {"Eliza", "Dmytro", "Mark"}

print(users & users2)   # {'Dmytro', 'Eliza'}
```

    {'Dmytro', 'Eliza'}
    

Модифікація методу - intersection_update() замінює пересіченими елементами першу множину:


```python
users = {"Andrii", "Dmytro", "Eliza"}
users2 = {"Eliza", "Dmytro", "Mark"}
users.intersection_update(users2)
print(users)   # {'Dmytro', 'Eliza'}
```

    {'Dmytro', 'Eliza'}
    

Ще одна операція - різниця множин повертає ті елементи, які є в першій множині, але відсутні в другій. Для отримання різниці множин можна використовувати метод difference або операцію віднімання:


```python
users = {"Andrii", "Dmytro", "Eliza"}
users2 = {"Eliza", "Dmytro", "Mark"}

users3 = users.difference(users2)
print(users3)           # {'Andrii'}
print(users - users2)   # {'Andrii'}
```

    {'Andrii'}
    {'Andrii'}
    

Окремий різновид різниці множин - симетрична різниця - проводиться за допомогою методу symmetric_difference() або за допомогою операції ^. Вона повертає всі елементи обох множин, за винятком спільних:


```python
users = {"Andrii", "Dmytro", "Eliza"}
users2 = {"Eliza", "Bob", "Mark"}

users3 = users.symmetric_difference(users2)
print(users3)   # {'Mark', 'Andrii', 'Bob', 'Dmytro'}

users4 = users ^ users2
print(users4)   # {'Mark', 'Andrii', 'Bob', 'Dmytro'}
```

    {'Bob', 'Mark', 'Andrii', 'Dmytro'}
    {'Bob', 'Mark', 'Andrii', 'Dmytro'}
    

Метод issubset дає змогу з'ясувати, чи є поточна множина підмножиною (тобто частиною) іншої множини:


```python
users = {"Tom", "Bob", "Alice"}
superusers = {"Sam", "Tom", "Bob", "Alice", "Greg"}

print(users.issubset(superusers))   # True
print(superusers.issubset(users))   # False
```

    True
    False
    

Метод issuperset, навпаки, повертає True, якщо поточна множина є надмножиною (тобто містить) для іншої множини:


```python
users = {"Tom", "Bob", "Alice"}
superusers = {"Sam", "Tom", "Bob", "Alice", "Greg"}

print(users.issuperset(superusers))   # False
print(superusers.issuperset(users))   # True
```

    False
    True
    

Тип frozen set є видом множин, який не може бути змінений. Для його створення використовується функція frozenset:


```python
users = frozenset({"Andrii", "Eliza", "Dmytro"})
```

У функцію frozenset передається набір елементів - список, кортеж, інша множина.

У таку множину ми не можемо додати нові елементи, як і видалити з неї вже наявні. Власне тому frozen set підтримує обмежений набір операцій:
1) len(s): повертає довжину множини

2) x in s: повертає True, якщо елемент x присутній у множині s

3) x not in s: повертає True, якщо елемент x відсутній у множині s

4) s.issubset(t): повертає True, якщо t містить множину s

5) s.issuperset(t): повертає True, якщо t міститься в множині s

6) s.union(t): повертає об'єднання множин s і t

7) s.intersection(t): повертає перетин множин s і t

8) s.difference(t): повертає різницю множин s і t

9) s.copy(): повертає копію множини s

# Словник
Словник (dictionary) у мові Python зберігає колекцію елементів, де кожен елемент має унікальний ключ і асоційоване з ним деяке значення.


dictionary = { ключ1:значення1, ключ2:значення2, ....}

У фігурних дужках через кому визначається послідовність елементів, де для кожного елемента спочатку вказується ключ і через двокрапку його значення.


```python
users = {1: "Andrii", 2: "Eliza", 3: "Dmytro"}
```

У словнику users як ключі використовуються числа, а як значення - рядки. Тобто елемент із ключем 1 має значення "Tom", елемент із ключем 2 - значення "Bob" тощо.

Інший приклад:


```python
emails = {"andrii@gmail.com": "Andrii", "eliza@gmai.com": "Eliza", "dmytro@gmail.com": "Dmytro"}
```

У словнику emails як ключі використовують рядки - електронні адреси користувачів і як значення теж рядки - імена користувачів.

Але необов'язково ключі та рядки мають бути однотипними. Вони можуть представляти різні типи:


```python
objects = {1: "Andrii", "2": True, 3: 100.6}
```

Ми можемо також взагалі визначити порожній словник без елементів:


```python
objects = {}
# або так
objects = dict()
```

Незважаючи на те, що словник і список - несхожі за структурою типи, проте існує можливість для окремих видів списків перетворення їх у словник за допомогою вбудованої функції dict(). Для цього список повинен зберігати набір вкладених списків. Кожен вкладений список повинен складатися з двох елементів - під час конвертації в словник перший елемент стане ключем, а другий - значенням:


```python
users_list = [
    ["+111123455", "Andrii"],
    ["+384767557", "Eliza"],
    ["+958758767", "Tom"]
]
users_dict = dict(users_list)
print(users_dict)      # {'+111123455': 'Andrii', '+384767557': 'Eliza', '+958758767': 'Tom'}
```

    {'+111123455': 'Andrii', '+384767557': 'Eliza', '+958758767': 'Tom'}
    

Подібним чином можна перетворити в словник двовимірні кортежі, які своєю чергою містять кортежі з двох елементів:


```python
users_tuple = (
    ("+111123455", "Tom"),
    ("+384767557", "Bob"),
    ("+958758767", "Alice")
)
users_dict = dict(users_tuple)
print(users_dict)
```

    {'+111123455': 'Tom', '+384767557': 'Bob', '+958758767': 'Alice'}
    

Для звернення до елементів словника після його назви у квадратних дужках вказується ключ елемента:
dictionary[ключ]


```python
users = {
    "+11111111": "Andrii",
    "+33333333": "Eliza",
    "+55555555": "Tom"
}

# отримуємо елемент із ключем "+1111111111"
print(users["+11111111"])      # Andrii

# встановлення значення елемента з ключем "+3333333333"
users["+33333333"] = "Bob Smith"
print(users["+33333333"])   # Bob Smith
```

    Andrii
    Bob Smith
    

Якщо під час встановлення значення елемента з таким ключем у словнику не виявиться, то відбудеться його додавання:


```python
users["+4444444"] = "Mark"
```

Але якщо ми спробуємо отримати значення з ключем, якого немає в словнику, то Python згенерує помилку KeyError:


```python
user = users["+4444444"]    # KeyError
```

І щоб попередити цю ситуацію перед зверненням до елемента ми можемо перевіряти наявність ключа в словнику за допомогою виразу ключ in словник. Якщо ключ є в словнику, то цей вираз повертає True:


```python
key = "+4444444"
if key in users:
    user = users[key]
    print(user)
else:
    print("Елемент не знайдено")
```

    Mark
    

Також для отримання елементів можна використовувати метод get, який має дві форми:

1) get(key): повертає зі словника елемент із ключем key. Якщо елемента з таким ключем немає, то повертає значення None

2) get(key, default): повертає зі словника елемент із ключем key. Якщо елемента з таким ключем немає, то повертає значення за замовчуванням default


```python
users = {
    "+11111111": "Andrii",
    "+33333333": "Eliza",
    "+55555555": "Dmytro"
}

user1 = users.get("+55555555")
print(user1)    # Dmytro
user2 = users.get("+33333333", "Unknown user")
print(user2)    # Eliza
user3 = users.get("+44444444", "Unknown user")
print(user3)    # Unknown user
```

    Dmytro
    Eliza
    Unknown user
    

Для видалення елемента за ключем застосовується оператор del:


```python
users = {
    "+11111111": "Andrii",
    "+33333333": "Eliza",
    "+55555555": "Dmytro"
}

del users["+55555555"]
print(users)    # {'+11111111': 'Andrii', '+33333333': 'Eliza'}
```

    {'+11111111': 'Andrii', '+33333333': 'Eliza'}
    

Але варто враховувати, що якщо такого ключа не виявиться в словнику, то буде викинуто виняток KeyError. Тому знову ж таки перед видаленням бажано перевіряти наявність елемента з цим ключем.


```python
users = {
    "+11111111": "Andrii",
    "+33333333": "Eliza",
    "+55555555": "Dmytro"
}

key = "+55555555"
if key in users:
    del users[key]
    print(f"Елемент із ключем {key} видалено")
else:
    print("Елемент не знайдено")
```

    Елемент із ключем +55555555 видалено
    

Інший спосіб видалення представляє метод pop(). Він має дві форми:

pop(key): видаляє елемент за ключем key і повертає видалений елемент. Якщо елемент з даним ключем відсутній, то генерується виняток KeyError

pop(key, default): видаляє елемент за ключем key і повертає видалений елемент. Якщо елемент з даним ключем відсутній, то повертається значення default


```python
users = {
    "+11111111": "Andrii",
    "+33333333": "Eliza",
    "+55555555": "Dmytro"
}
key = "+55555555"
user = users.pop(key)
print(user)     # Dmytro

user = users.pop("+4444444", "Unknown user")
print(user)     # Unknown user
```

Якщо необхідно видалити всі елементи, то в цьому разі можна скористатися методом clear():


```python
users.clear()
```

Метод copy() копіює вміст словника, повертаючи новий словник:


```python
users = {"+1111111": "Andrii", "+3333333": "Eliza", "+5555555": "Dmytro"}
students = users.copy()
print(students)     # {'+1111111': 'Andrii', '+3333333': 'Eliza', '+5555555': 'Dmytro'}
```

    {'+1111111': 'Andrii', '+3333333': 'Eliza', '+5555555': 'Dmytro'}
    

Метод update() об'єднує два словники:


```python
users = {"+1111111": "Andrii", "+3333333": "Eliza"}

users2 = {"+2222222": "Dmytro", "+6666666": "Mark"}
users.update(users2)

print(users)    # {'+1111111': 'Andrii', '+3333333': 'Eliza', '+2222222': 'Dmytro', '+6666666': 'Mark'}
print(users2)   # {'+2222222': 'Dmytro', '+6666666': 'Mark'}
```

    {'+1111111': 'Andrii', '+3333333': 'Eliza', '+2222222': 'Dmytro', '+6666666': 'Mark'}
    {'+2222222': 'Dmytro', '+6666666': 'Mark'}
    

При цьому словник users2 залишається без змін. Змінюється тільки словник users, до якого додаються елементи іншого словника. Але якщо необхідно, щоб обидва вихідні словники були без змін, а результатом об'єднання був якийсь третій словник, то можна попередньо скопіювати один словник в інший:


```python
users3 = users.copy()
users3.update(users2)
```

Для перебору словника можна скористатися циклом for:


```python
users = {
    "+11111111": "Andrii",
    "+33333333": "Eliza",
    "+55555555": "Dmytro"
}
for key in users:
    print(f"Номер телефону: {key}  Користувач: {users[key]} ")
```

    Номер телефону: +11111111  Користувач: Andrii 
    Номер телефону: +33333333  Користувач: Eliza 
    Номер телефону: +55555555  Користувач: Dmytro 
    

Під час перебору елементів ми отримуємо ключ поточного елемента і за ним можемо отримати сам елемент.

Інший спосіб перебору елементів - використання методу items():


```python
users = {
    "+11111111": "Andrii",
    "+33333333": "Eliza",
    "+55555555": "Dmytro"
}
for key, value in users.items():
    print(f"Номер телефону: {key}  Користувач: {value} ")
```

    Номер телефону: +11111111  Користувач: Andrii 
    Номер телефону: +33333333  Користувач: Eliza 
    Номер телефону: +55555555  Користувач: Dmytro 
    

Метод items() повертає набір кортежів. Кожен кортеж містить ключ і значення елемента, які під час перебору ми тут же можемо отримати у змінні key і value.

Також існують окремо можливості перебору ключів і перебору значень. Для перебору ключів ми можемо викликати у словника метод keys():


```python
for key in users.keys():
    print(key)
```

    +11111111
    +33333333
    +55555555
    

Щоправда, цей спосіб перебору не має сенсу, оскільки і без виклику методу keys() ми можемо перебрати ключі, як було показано вище.

Для перебору тільки значень ми можемо викликати у словника метод values():


```python
for value in users.values():
    print(value)
```

    Andrii
    Eliza
    Dmytro
    

Крім найпростіших об'єктів на кшталт чисел і рядків словники також можуть зберігати і більш складні об'єкти - ті ж списки, кортежі або інші словники:


```python
users = {
    "Andrii": {
        "phone": "+971478745",
        "email": "andrii12@gmail.com"
    },
    "Eliza": {
        "phone": "+876390444",
        "email": "eliza@gmail.com",
        "skype": "eliza123"
    }
}
```

У цьому випадку значення кожного елемента словника своєю чергою представляє окремий словник.

Для звернення до елементів вкладеного словника відповідно необхідно використовувати два ключі:


```python
old_email = users["Andrii"]["email"]
users["Eliza"]["email"] = "elizabeth@gmail.com"
print(users["Eliza"])     # {'phone': '+876390444', 'email': 'elizabeth@gmail.com', 'skype': 'eliza123'}
```

    {'phone': '+876390444', 'email': 'elizabeth@gmail.com', 'skype': 'eliza123'}
    

Але якщо ми спробуємо отримати значення за ключем, який відсутній у словнику, Python згенерує виняток KeyError:


```python
tom_skype = users["Tom"]["skype"] # KeyError
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-115-3b126523702e> in <cell line: 1>()
    ----> 1 tom_skype = users["Tom"]["skype"] # KeyError
    

    KeyError: 'Tom'


Щоб уникнути помилки, можна перевіряти наявність ключа в словнику:


```python
key = "skype"
if key in users["Tom"]:
    print(users["Tom"]["skype"])
else:
    print("skype is not found")
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-116-362e6fcedcad> in <cell line: 2>()
          1 key = "skype"
    ----> 2 if key in users["Tom"]:
          3     print(users["Tom"]["skype"])
          4 else:
          5     print("skype is not found")
    

    KeyError: 'Tom'


У всьому іншому робота з комплексними і вкладеними словниками аналогічна роботі зі звичайними словниками.
