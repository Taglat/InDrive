## Инструкции по запуску в Docker

1. **Собрать Docker образ:**

```bash
docker build -t car-visual-check .
```

2. **Скачивание датасетов:**

```bash
docker run --rm -e ROBOFLOW_API_KEY="ваш_ключ" `
           -v ${PWD}/data:/app/data `
           car-visual-check download_datasets.py  
```

3. **Подготовка данных:**
```bash
docker run --rm -e ROBOFLOW_API_KEY="ваш_ключ" `
           -v ${PWD}/data:/app/data `
           -v ${PWD}/prepared:/app/prepared_data `
           car-visual-check prepare_data.py
```

4. **Гибкость:**

* Можно запускать любой скрипт в контейнере, например:

```bash
docker run -e ROBOFLOW_API_KEY="ключ" -v $(pwd)/data:/app/data car-visual-check another_script.py
```

---