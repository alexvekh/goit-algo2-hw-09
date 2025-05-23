# Алгоритми оптимізації: Hill Climbing, Random Local Search, Simulated Annealing

Цей проєкт реалізує та порівнює три популярні алгоритми оптимізації для мінімізації **функції Сфери**:

- Hill Climbing
- Random Local Search
- Simulated Annealing

## 📌 Функція цілі

Функція Сфери — це стандартна тестова функція у задачах оптимізації:

        f(x, y) = x² + y²

Мінімум цієї функції досягається в точці (0, 0), де `f(x, y) = 0`.

---

## 📦 Структура проєкту

- `main.py` — основний файл з реалізацією алгоритмів та візуалізацією результатів.
- `README.md` — цей файл.

---

## 🚀 Запуск

Виконайте файл `main.py`:

```bash
python main.py
```

## 🧠 Реалізовані алгоритми

- **Hill Climbing** - Покроково рухається до кращого сусіда. Зупиняється, коли немає покращення.
- **Random Local Search** - На кожному кроці випадковим чином шукає нову точку поблизу поточної. Іноді приймає гірші рішення, щоб уникнути локального мінімуму.
- **Simulated Annealing** - Імітує процес охолодження матеріалу. З великою ймовірністю приймає гірші рішення на початку, але поступово зменшує "температуру", щоб стабілізуватись у мінімумі.

## 📊 Візуалізація
Використовується matplotlib для побудови графіка рівнів (contour plot), де відображаються точки, знайдені кожним з алгоритмів.
На гафіку, для кращого розпізнання, зображено наближений центр сфери [-1, 1], [-1, 1]

![Візуалізація](image.png)

# Результати:

- Функція Сфери є простою, але ефективною тестовою функцією для демонстрації роботи алгоритмів оптимізації. Її глобальний мінімум легко знайти, що дозволяє добре ілюструвати поведінку методів. - Візуально видно, як близько кожен алгоритм знаходиться до глобального мінімуму.



- **Hill Climbing:**
    - ~0.000000000000000000000000000002
    - Найкращий результат.
    - Дійшов майже точно до глобального мінімуму (0, 0).
    - Працює швидко і просто.
    - Однак, якщо функція мала б багато локальних мінімумів, він би легко застряг у них.
    - Не робить "випадкових" кроків, тому не міг би вийти з плато або ямки.

- **Random Local Search:**
    - ~0.00927
    - Найгірше наближення серед трьох.
    - Випадкові кроки не привели до достатньо хорошого результату цього разу.
    - Може працювати краще при більшій кількості ітерацій або налаштуванні параметрів.
    - Краще уникає локальних мінімумів завдяки випадковим крокам.
    - Результат не завжди стабільний — залежить від ймовірності переходу та початкової точки.
    - Може знайти глобальний мінімум, але не гарантує цього.

- **Simulated Annealing:**
    - ~0.0000325
    - Займає проміжне місце.
    - Дає хороший результат, бо під час високої "температури" здатен виходити з локального мінімуму, а завдяки "охолодженню" знаходить самий мінімум.
    - Може перевершити Hill Climbing при достатній кількості запусків або налаштуванні температури.
    - Найгнучкіший та найпотужніший з усіх трьох.
    - Під час .
    - Повільно, але стабільно знаходить близьке до оптимального рішення.

## 🏁 Висновки:
- Якщо швидкість важливіша за універсальність і немає складної топології — Hill Climbing працює чудово.
- Якщо задача має багато локальних мінімумів, або початкова точка неідеальна — Simulated Annealing дає більш стабільні результати.
- Random Local Search є простим, але менш ефективним без ретельного налаштування параметрів.













## 🔒 Обмеження
Кожен алгоритм працює в межах: x ∈ [-5, 5], y ∈ [-5, 5]
Точки за межі не виходять завдяки функції clamp.

## 🧩 Залежності
- numpy
- matplotlib
Встановіть їх (якщо потрібно):

        pip install numpy matplotlib

## 📚 Автор 
Цей проєкт реалізований в рамках навчання в GoIT.