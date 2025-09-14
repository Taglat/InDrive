
docker build --no-cache -t car-visual-check .

docker run --rm -e ROBOFLOW_API_KEY="3qCQ6UK3ceg9YNx9I3lp" `
           -v ${PWD}/data:/app/data `
           car-visual-check python download_datasets.py

docker run --rm -e ROBOFLOW_API_KEY="3qCQ6UK3ceg9YNx9I3lp" `
           -v ${PWD}/data:/app/data `
           -v ${PWD}/prepared:/app/prepared `
           car-visual-check python prepare_data.py

docker run --rm -v ${PWD}/prepared:/app/prepared `
           car-visual-check python train_classifier.py