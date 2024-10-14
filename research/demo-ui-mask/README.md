# **Indentification of bowhead whales**

Модели: https://drive.google.com/drive/folders/1PgaDAW-YwJW4xzzq41F0_2-bdn18jCbd

Реализовано веб-приложения для индетификации гренландских китов с бинарной маской. Для отслеживания миграции и популяции семейства.

### **Установка и запуск**

Clone the repo and change to the project root directory:

Install requirements:
```
pip install -r requirements.txt
```

And run:
```
streamlit run streamlit_app.py
```

### **Пример работы**

![demo](./resources/whales_demo.gif)


## **Используемое решение**

Технологии для решения отбирались по двум критериям - это автономность работы решения и наличие библиотек открытого доступа. 
После чего мы провели сравнительный анализ, выбрав модели оптимальные по соотношению точности к требуемой производительности - ResNet101, VGG19 и EfficientNet.
