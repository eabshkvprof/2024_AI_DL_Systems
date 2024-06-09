# Лекція 6.3. Файлові об'єкти. Визначення загальної структури програми

# Файли
Python підтримує безліч різних типів файлів, але умовно їх можна розділити на два види: текстові та бінарні. Текстові файли - це, наприклад, файли з розширенням csv, txt, html, загалом будь-які файли, які зберігають інформацію в текстовому вигляді. Бінарні файли - це зображення, аудіо та відеофайли тощо. Залежно від типу файлу робота з ним може трохи відрізнятися.

Під час роботи з файлами необхідно дотримуватися певної послідовності операцій:
1.   Відкриття файлу за допомогою методу open()
2.   Читання файлу за допомогою методу read() або запис у файл за допомогою методу write()
3. Закриття файлу методом close()

Щоб почати роботу з файлом, його треба відкрити за допомогою функції open(), яка має таке формальне визначення:

```
open(file, mode)
```

Перший параметр функції представляє шлях до файлу. Шлях файлу може бути абсолютним, тобто починатися з букви диска, наприклад, C://somedir/somefile.txt. Або може бути відносним, наприклад, somedir/somefile.txt - у цьому разі пошук файлу буде йти щодо розташування запущеного скрипта Python.
Другий переданий аргумент - mode встановлює режим відкриття файлу залежно від того, що ми збираємося з ним робити. Існує 4 загальних режими:

1. r (Read). Файл відкривається для читання. Якщо файл не знайдено, то генерується виняток FileNotFoundError

2. w (Write). Файл відкривається для запису. Якщо файл відсутній, то він створюється. Якщо подібний файл уже є, то він **створюється заново**, і відповідно старі дані в ньому **стираються**.

3. a (Append). Файл відкривається для дозапису. Якщо файл відсутній, то він створюється. Якщо подібний файл уже є, то дані записуються в його кінець.

4. b (Binary). Використовується для роботи з бінарними файлами. Застосовується разом з іншими режимами - w або r.

Після завершення роботи з файлом його обов'язково потрібно закрити методом close(). Цей метод звільнить усі пов'язані з файлом ресурси, що використовуються.


```python
# Для прикладу відкриємо тектовий файл для запису
myfile = open("hello.txt", "w")

myfile.close()
```

Під час відкриття файлу або в процесі роботи з ним ми можемо зіткнутися з різними винятками, наприклад, до нього немає доступу тощо. У цьому випадку програма випаде в помилку, а її виконання не дійде до виклику методу close, і відповідно файл не буде закрито.

У цьому випадку ми можемо обробляти винятки:


```python
try:
    somefile = open("hello.txt", "w")
    try:
        somefile.write("hello world")
    except Exception as e:
        print(e)
    finally:
        somefile.close()
except Exception as ex:
    print(ex)
```

У цьому разі вся робота з файлом йде у вкладеному блоці try. І якщо раптом виникне якесь виключення, то в будь-якому разі в блоці finally файл буде закрито.

Однак є і більш зручна конструкція - конструкція with:


```python
with open("hello.txt", "w") as file_obj:
  file_obj.write("hello world")
```

Ця конструкція визначає для відкритого файлу змінну file_obj і виконує набір інструкцій. Після їх виконання файл автоматично закривається. Навіть якщо під час виконання інструкцій у блоці with виникнуть будь-які винятки, то файл однаково закривається.

Щоб відкрити текстовий файл на запис, необхідно застосувати режим w (перезапис) або a (дозапис). Потім для запису застосовується метод write(str), у який передається записуваний рядок. Варто зазначити, що записується саме рядок, тому, якщо потрібно записати числа, дані інших типів, то їх попередньо потрібно конвертувати в рядок.

Запишемо деяку інформацію у файл "hello.txt":


```python
with open("hello.txt", "a") as file:
    file.write("hello world")
```

Якщо ми відкриємо папку, в якій знаходиться поточний скрипт Python, то побачимо там файл hello.txt. Цей файл можна відкрити в будь-якому текстовому редакторі та за бажання змінити.

Тепер дозапишемо в цей файл ще один рядок:


```python
with open("hello.txt", "a") as file:
    file.write("\ngood bye, world")
```

Дозапис виглядає як додавання рядка до останнього символу у файлі, тому, якщо необхідно зробити запис з нового рядка, то можна використовувати ескейп-послідовність "\n".

Ще один спосіб запису до файлу представляє стандартний метод print(), який застосовується для виведення даних на консоль:


```python
with open("hello.txt", "a") as hello_file:
    print("Hello, world", file=hello_file)
```

Для виведення даних у файл у метод print як другий параметр передається назва файлу через параметр file. А перший параметр представляє рядок, що записується у файл.

Для читання файлу він відкривається з режимом r (Read), і потім ми можемо зчитати його вміст різними методами:
1. readline(): зчитує один рядок із файлу
2. read(): зчитує весь вміст файлу в один рядок
3. readlines(): зчитує всі рядки файлу в список

Наприклад, зчитаємо вище записаний файл порядково:


```python
with open("hello.txt", "r") as file:
    for line in file:
        print(line, end="")
```

    hello worldhello worldhello world
    Hello, world
    

Незважаючи на те, що ми явно не застосовуємо метод readline() для читання кожного рядка, але під час перебору файлу цей метод автоматично викликається для отримання кожного нового рядка. Тому в циклі вручну немає сенсу викликати метод readline. І оскільки рядки розділяються символом переведення рядка "\n", то щоб унеможливити зайвого перенесення на інший рядок у функцію print передається значення end="".

Тепер явним чином викличемо метод readline() для читання окремих рядків:


```python
with open("hello.txt", "r") as file:
    str1 = file.readline()
    print(str1, end="")
    str2 = file.readline()
    print(str2)
```

    hello worldhello worldhello world
    Hello, world
    
    

Метод readline можна використовувати для рядкового зчитування файлу в циклі while:


```python
with open("hello.txt", "r") as file:
    line = file.readline()
    while line:
        print(line, end="")
        line = file.readline()
```

    hello worldhello worldhello world
    Hello, world
    

Якщо файл невеликий, то його можна разом зчитати за допомогою методу read():


```python
with open("hello.txt", "r") as file:
    content = file.read()
    print(content)
```

    hello worldhello worldhello world
    Hello, world
    
    

І також застосуємо метод readlines() для зчитування всього файлу в список рядків:


```python
with open("hello.txt", "r") as file:
    contents = file.readlines()
    str1 = contents[0]
    str2 = contents[1]
    print(str1, end="")
    print(str2)
```

    hello world
    good bye, worldHello, world
    
    

Під час читання файлу ми можемо зіткнутися з тим, що його кодування не збігається з ASCII. У цьому випадку ми явно можемо вказати кодування за допомогою параметра encoding:


```python
filename = "hello.txt"
with open(filename, encoding="utf8") as file:
    text = file.read()
```

Тепер напишемо невеликий скрипт, у якому записуватиме введений користувачем масив рядків і зчитуватиме його назад із файлу на консоль:


```python
filename="messages.txt"
messages=list()

for i in range(4):
  message = str(input("Enter string: " + str(i+1) + ":"))
  messages.append(message)

with open("messages.txt","w") as file:
  for message in messages:
    file.write(message+"\n")

print("Read data")
with open("messages.txt","r") as file:
  for message in file:
    print(message, end="")
```

    Enter string: 1:Andrii
    Enter string: 2:Nikitenko
    Enter string: 3:Slava
    Enter string: 4:Ukraine
    Read data
    Andrii
    Nikitenko
    Slava
    Ukraine
    

# Файли CSV

Одним із поширених файлових форматів, які зберігають у зручному вигляді інформацію, є формат csv. Кожен рядок у файлі csv представляє окремий запис або рядок, який складається з окремих стовпців, розділених комами. Власне тому формат і називається Comma Separated Values. Але хоча формат csv - це формат текстових файлів, Python для спрощення роботи з ним надає спеціальний вбудований модуль csv.

У прикладі нижче у файл записується двовимірний список - фактично таблиця, де кожен рядок представляє одного користувача. А кожен користувач містить два поля - ім'я та вік. Тобто фактично таблиця з трьох рядків і двох стовпців.

Під час відкриття файлу на запис як третій параметр вказується значення newline=""" - порожній рядок дає змогу коректно зчитувати рядки з файлу незалежно від операційної системи.

Для запису нам треба отримати об'єкт writer, який повертається функцією csv.writer(file). У цю функцію передається відкритий файл. А власне запис здійснюється за допомогою методу writer.writerows(users) Цей метод приймає набір рядків. У нашому випадку це двовимірний список.

Якщо необхідно додати один запис, який являє собою одновимірний список, наприклад, ["Sam", 31], то в цьому випадку можна викликати метод writer.writerow(user)

У підсумку після виконання скрипта в тій самій папці опиниться файл users.csv, який матиме такий вміст:


```python
import csv

FILENAME = "users.csv"

users = [
    ["Tom", 28],
    ["Alice", 23],
    ["Bob", 34]
]

with open(FILENAME, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(users)


with open(FILENAME, "a", newline="") as file:
    user = ["Sam", 31]
    writer = csv.writer(file)
    writer.writerow(user)
```

Для читання з файлу нам навпаки потрібно створити об'єкт reader:


```python
import csv

FILENAME = "users.csv"

with open(FILENAME, "r", newline="") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row[0], " - ", row[1])
```

    Tom  -  28
    Alice  -  23
    Bob  -  34
    Sam  -  31
    

## Робота зі словниками

У прикладі вище кожен запис або рядок являв собою окремий список, наприклад, ["Sam", 31]. Але крім того, модуль csv має спеціальні додаткові можливості для роботи зі словниками. Зокрема, функція csv.DictWriter() повертає об'єкт writer, який дає змогу записувати у файл. А функція csv.DictReader() повертає об'єкт reader для читання з файлу. Наприклад:


```python
import csv

FILENAME = "users.csv"

users = [
    {"name": "Tom", "age": 28},
    {"name": "Alice", "age": 23},
    {"name": "Bob", "age": 34}
]

with open(FILENAME, "w", newline="") as file:
    columns = ["name", "age"]
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()

    # запис декількох рядків
    writer.writerows(users)

    user = {"name" : "Sam", "age": 41}
    # запис одного рядка
    writer.writerow(user)

with open(FILENAME, "r", newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row["name"], "-", row["age"])
```

    Tom - 28
    Alice - 23
    Bob - 34
    Sam - 41
    

Запис рядків також здійснюється за допомогою методів writerow() і writerows(). Але тепер кожен рядок являє собою окремий словник, і крім того, проводиться запис і заголовків стовпців за допомогою методу writeheader(), а в метод csv.DictWriter як другий параметр передається набір стовпців.

Під час читання рядків, використовуючи назви стовпців, ми можемо звернутися до окремих значень усередині рядка: row["name"].

# Бінарні файли

Бінарні файли на відміну від текстових зберігають інформацію у вигляді набору байт. Для роботи з ними в Python необхідний вбудований модуль pickle. Цей модуль надає два методи:
1. dump(obj, file): записує об'єкт obj у бінарний файл file
2. load(file): зчитує дані з бінарного файлу в об'єкт

Під час відкриття бінарного файлу на читання або запис також треба враховувати, що нам потрібно застосовувати режим "b" на додачу до режиму запису ("w") або читання ("r"). Припустимо, треба треба зберегти два об'єкти:


```python
import pickle

FILENAME = "user.dat"

name = "Tom"
age = 19

with open(FILENAME, "wb") as file:
    pickle.dump(name, file)
    pickle.dump(age, file)

with open(FILENAME, "rb") as file:
    name = pickle.load(file)
    age = pickle.load(file)
    print("Ім'я:", name, "\tВік:", age)
```

    Ім'я: Tom 	Вік: 19
    

За допомогою функції dump послідовно записуються два об'єкти. Тому під час читання файлу також послідовно за допомогою функції load ми можемо зчитати ці об'єкти.

Подібним чином ми можемо зберігати і витягувати з файлу набори об'єктів:


```python
import pickle

FILENAME = "users.dat"

users = [
    ["Tom", 28, True],
    ["Alice", 23, False],
    ["Bob", 34, False]
]

with open(FILENAME, "wb") as file:
    pickle.dump(users, file)


with open(FILENAME, "rb") as file:
    users_from_file = pickle.load(file)
    for user in users_from_file:
        print("Name:", user[0], "\tAge:", user[1], "\tMarried or not:", user[2])
```

    Name: Tom 	Age: 28 	Married or not: True
    Name: Alice 	Age: 23 	Married or not: False
    Name: Bob 	Age: 34 	Married or not: False
    

Залежно від того, який об'єкт ми записували функцією dump, той самий об'єкт буде повернуто функцією load під час зчитування файлу.

## Модуль OS і робота з файловою системою



Низку можливостей щодо роботи з каталогами та файлами надає вбудований модуль os. Хоча він містить багато функцій, розглянемо тільки основні з них:
1. mkdir(): створюємо нову папку
2. rmdir(): видаляємо папку
3. rename(): змінюємо назву файлу
4. remove(): видаляємо файл

Для створення папки застосовується функція mkdir(), в яку передається шлях до створюваної папки:


```python
import os

# шлях відносно поточного скрипта
os.mkdir("шлях відносно поточного скрипта")
# абсолютний шлях
os.mkdir("/content/шлях відносно поточного скрипта/dir1")
os.mkdir("/content/шлях відносно поточного скрипта/dir2")
```

Для видалення папки використовується функція rmdir(), в яку передано шлях до папки, що видаляється:


```python
import os

# шлях відносно поточного скрипта
os.rmdir("/content/шлях відносно поточного скрипта/dir2")
# абсолютний шлях
# os.rmdir("c://somedir/hello")
```

Для перейменування викликається функція rename(source, target), перший параметр якої - шлях до вихідного файлу, а другий - нове ім'я файлу. Як шляхи можуть використовуватися як абсолютні, так і відносні. Наприклад, нехай у папці C://SomeDir/ розташовується файл somefile.txt. Перейменуємо його на файл "hello.txt":


```python
import os

os.rename("/content/шлях відносно поточного скрипта/dir1", "/content/шлях відносно поточного скрипта/dir2")
```

Для видалення викликається функція remove(), в яку передається шлях до файлу:


```python
import os

os.remove("/content/hello.txt")
```

Якщо ми спробуємо відкрити файл, який не існує, то Python викине виняток FileNotFoundError. Для відлову винятку ми можемо використовувати конструкцію try...except. Однак можна вже до відкриття файлу перевірити, існує він чи ні за допомогою методу os.path.exists(path). У цей метод передається шлях, який необхідно перевірити:


```python
filename = input("Введите путь к файлу: ")
if os.path.exists(filename):
    print("Зазначений файл існує")
else:
    print("Файл не існує")
```

    Введите путь к файлу: /content/user.da
    Файл не існує
    

## Запис і читання архівних zip-файлів

Zip представляє найбільш популярний формат архівації та стиснення файлів. І мова Python має вбудований модуль для роботи з ними - zipfile. За допомогою цього модуля можна створювати, зчитувати, записувати zip-файли, отримувати їхній вміст і додавати в них файли. Також підтримується шифрування, але не підтримується дешифрування.

Для представлення zip-файлу в цьому модулі визначено клас ZipFile. Він має такий конструктор:


```
ZipFile(file, mode='r', compression=ZIP_STORED, allowZip64=True, compresslevel=None, *, strict_timestamps=True, metadata_encoding=None)
```



Параметри:
*   file: шлях до zip-файлу
*   mode: режим відкриття файлу. Може набувати таких значень:
 * r: застосовується для читання наявного файлу

 * w: застосовується для запису нового файлу

 * a: застосовується для додавання у файл

* compression: тип стиснення файлу під час запису. Може набувати значень:

 * ZIP_STORED: архівація без стиснення (значення за замовчуванням)

 * ZIP_DEFLATED: стандартний тип стиснення під час архівації в zip

 * ZIP_BZIP2: стиснення за допомогою способу BZIP2

 * ZIP_LZMA: стиснення за допомогою способу LZMA

* allowZip64: якщо дорівнює True, то zip-файл може бути більшим за 4 Гб

* compresslevel: рівень стиснення під час запису файлу. Для типів стиснення ZIP_STORED і ZIP_LZMA не застосовується. Для типу ZIP_DEFLATED допустимі значення від 0 до 9, а для типу ZIP_BZIP2 допустимі значення від 1 до 9.

* strict_timestamps: при значенні False дає змогу працювати з zip-файлами, створеними раніше 01.01.1980 і пізніше 31.12.2107

* metadata_encoding: застосовується для декодування метаданих zip-файлу (наприклад, коментарів)

Для роботи з файлами цей клас надає низку методів (функції):

* close(): закриває zip-файл
* getinfo(): повертає інформацію про один файл з архіву у вигляді об'єкта ZipInfo
* namelist(): повертає список файлів архіву
* infolist(): повертає інформацію про всі файли з архіву у вигляді списку об'єктів ZipInfo
* open(): надає доступ до одного з файлів в архіві
* read(): зчитує файл з архіву в набір байтів
* extract(): витягує з архіву один файл
* extractall(): витягує всі елементи з архіву
* setpassword(): встановлює пароль для zip-файлу
* printdir(): виводить на консоль вміст архіву

Для створення архівного файлу в конструктор ZipFile передається режим "w" або "a":


```python
from zipfile import ZipFile

myzip = ZipFile("andrii.zip", "w")
```

Після виконання коду в поточній папці буде створюватися порожній архівний файл "andrii.zip".
Після закінчення роботи з архівом для його закриття застосовується метод close():


```python
from zipfile import ZipFile

myzip = ZipFile("andrii.zip", "w")
myzip.close()
```

Але оскільки ZipFile також представляє менеджер контексту, то він підтримує вираз with, який визначає контекст і автоматично закриває файл після завершення контексту:


```python
from zipfile import ZipFile

with ZipFile("andrii.zip", "w") as myzip:
    pass
```

Для запису файлів в архів застосовується файл write():


```
write(filename, arcname=None, compress_type=None, compresslevel=None)
```



Перший параметр представляє файл, який записується в архів. Другий параметр - arcname встановлює довільне ім'я для файлу всередині архіву (за замовчуванням це саме ім'я файлу). Третій параметр - compress_type представляє тип стиснення, а параметр compresslevel - рівень стиснення.

Наприклад, запишемо в архів "andrii.zip" файл "hello.txt" (який, як припускається, знаходиться в тій самій папці, де і поточний скрипт python):


```python
from zipfile import ZipFile

with ZipFile("andrii.zip", "w") as myzip:
    myzip.write("users.csv")
```

Варто враховувати, що під час відкриття файлу в режимі "w" під час усіх наступних записів поточний вміст буде затиратися, тобто фактично архівний файл буде створюватися заново. Якщо нам необхідно додати, то необхідно визначати zip-файл у режимі "a":


```python
from zipfile import ZipFile

with ZipFile("andrii.zip", "a") as myzip:
    myzip.write("users.dat")
```

Варто зазначити, що за замовчуванням стиснення не застосовується. Але за необхідності можна застосувати який-небудь спосіб стиснення і рівень стиснення"


```python
from zipfile import ZipFile, ZIP_DEFLATED

with ZipFile("andrii.zip", "w", compression=ZIP_DEFLATED, compresslevel=3) as myzip:
    myzip.write("user.dat")
```

Необхідно враховувати, що якщо ми спробуємо додати в архів файли з уже наявними іменами, то консоль виведе попередження. Щоб уникнути наявності файлів з іменами, що дублюються, можна через другий параметр методу write явним чином визначити для них унікальне ім'я всередині архіву:


```python
from zipfile import ZipFile

with ZipFile("andrii.zip", "a") as myzip:
    myzip.write("hello.txt", "hello1.txt")
    myzip.write("hello.txt", "hello2.txt")
    myzip.write("hello.txt", "hello3.txt")
```

Метод infolist() повертає інформацію про файли в архіві у вигляді списку, де кожен окремий файл представлений об'єктом ZipInfo:


```python
from zipfile import ZipFile

with ZipFile("andrii.zip", "a") as myzip:
    print(myzip.infolist())
```

    [<ZipInfo filename='user.dat' compress_type=deflate filemode='-rw-r--r--' file_size=23 compress_size=20>]
    

Клас ZipInfo надає низку атрибутів для зберігання інформації про файл. Основні з них:
1. filename: назва файлу
2. date_time: дата і час останньої зміни файлу у вигляді кортежу у форматі (рік, місяць, день, година, хвилина, секунда)
3. compress_type: тип стиснення
4. compress_size: розмір після стиснення
5. file_size: оригінальний розмір файлу до стиснення

Отримаємо ці дані за кожним окремим файлом в архіві:


```python
from zipfile import ZipFile

with ZipFile("andrii.zip", "r") as myzip:
    for item in myzip.infolist():
        print(f"File Name: {item.filename} Date: {item.date_time} Size: {item.file_size}")
```

    File Name: user.dat Date: (2024, 3, 4, 10, 1, 48) Size: 23
    

За допомогою методу is_dir() можна перевірити, чи є елемент в архіві папкою:


```python
from zipfile import ZipFile

with ZipFile("andrii.zip", "r") as myzip:
    for item in myzip.infolist():
        if(item.is_dir()):
            print(f"Папка: {item.filename}")
        else:
            print(f"Файл: {item.filename}")
```

    Файл: user.dat
    

Якщо треба отримати тільки список імен файлів, що входять в архів, то застосовується метод namelist():


```python
from zipfile import ZipFile

with ZipFile("andrii.zip", "r") as myzip:
    for item in myzip.namelist():
        print(item)
```

    user.dat
    

За допомогою методу getinfo() можна отримати дані по одному з архівованих файлів, передавши в метод його ім'я в архіві. Результат методу - об'єкт ZipInfo:


```python
from zipfile import ZipFile

with ZipFile("andrii.zip", "r") as myzip:
    try:
        hello_file = myzip.getinfo("user.dat")
        print(hello_file.file_size)
    except KeyError:
        print("Зазначений файл відсутній")
```

    23
    

Якщо в архіві не виявиться елемента із зазначеним ім'ям, то метод згенерує помилку KeyError.

Для вилучення всіх файлів з архіву застосовується метод extractall():


```
extractall(path=None, members=None, pwd=None)
```




Перший параметр методу встановлює каталог для вилучення архіву (за замовчуванням вилучення йде в поточний каталог). Параметр members представляє список рядків - список назв файлів, які треба витягти з архіву. І третій параметр - pwd представляє пароль, у разі якщо архів закритий паролем.

Наприклад, витягнемо всі файли з архіву:


```python
from zipfile import ZipFile

with ZipFile("andrii.zip", "r") as myzip:
    myzip.extractall()
```

Витяг у певну папку:


```python
with ZipFile("andrii.zip", "r") as myzip:
    myzip.extractall(path="andrii_folder")
```

Вилучення частини файлів:


```python
# витягуємо файли  "hello.txt", "forest.jpg" в папку "andrii_folder_2"
myzip.extractall(path="metanit2", members=["hello.txt", "forest.jpg"])
```

Для вилучення одного файлу застосовується метод extract(), у який як обов'язковий параметр передають ім'я файлу, що витягується:


```python
myzip.extract("hello.txt")
```

Метод read() дає змогу зчитати вміст файлу з архіву в набір байтів:


```python
from zipfile import ZipFile

with ZipFile("andrii.zip", "r") as myzip:
    content = myzip.read("hello5.txt")
    print(content)
```

Метод open() дає змогу відкривати окремі файли з архіву без безпосереднього їх вилучення:

```
open(name, mode='r', pwd=None, *, force_zip64=False)
```

Як перший обов'язковий параметр передається ім'я файлу всередині архіву. Другий параметр - mode встановлює режим відкриття. Параметр pwd задає пароль, якщо файл захищений паролем. І параметр force_zip64 при значенні True дає змогу відкривати файли більше 4 Гб.

Цей файл може бути корисний для маніпулювання файлом, наприклад, для зчитування його вмісту або, навпаки, для запису в нього. Наприклад, відкриємо файл і зчитаємо його вміст:





```python
from zipfile import ZipFile

with ZipFile("andrii.zip", "a") as myzip:
    # записуємо в архів новий файл "hello5.txt"
    with myzip.open("hello5.txt", "w") as hello_file:
        encoded_str = bytes("Python...", "UTF-8")
        hello_file.write(encoded_str)
```

# Example of docstring (function)


```python
def calculate_area_of_rectangle(length, width):
    '''
    Returns the area of rectangle

            Parameters:
                    length (float): The length of the rectangle.
                    width (float): The width of the rectangle.

            Returns:
                    float: The area of the rectangle.
    '''
    area = length * width
    return area
```
