#!/usr/bin/env python3
"""
Скрипт для запуска веб-приложения анализа повреждений автомобилей
"""

import os
import sys
import subprocess
import argparse

def check_dependencies():
    """Проверяет установлены ли необходимые зависимости"""
    try:
        import flask
        import torch
        import torchvision
        from PIL import Image
        print("✅ Все зависимости установлены")
        return True
    except ImportError as e:
        print(f"❌ Отсутствует зависимость: {e}")
        print("Установите зависимости командой: pip install -r requirements.txt")
        return False

def check_model():
    """Проверяет наличие обученной модели"""
    model_path = "prepared/best_model.pth"
    if os.path.exists(model_path):
        print(f"✅ Модель найдена: {model_path}")
        return True
    else:
        print(f"❌ Модель не найдена: {model_path}")
        print("Сначала обучите модель командой: python train_classifier.py")
        return False

def install_dependencies():
    """Устанавливает зависимости"""
    print("Устанавливаем зависимости...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Зависимости установлены")
        return True
    except subprocess.CalledProcessError:
        print("❌ Ошибка при установке зависимостей")
        return False

def run_app(host="0.0.0.0", port=5000, debug=False):
    """Запускает веб-приложение"""
    print(f"🚀 Запускаем веб-приложение на http://{host}:{port}")
    print("Для остановки нажмите Ctrl+C")
    
    # Устанавливаем переменные окружения для Flask
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development' if debug else 'production'
    
    try:
        from app import app
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n👋 Веб-приложение остановлено")
    except Exception as e:
        print(f"❌ Ошибка при запуске: {e}")

def main():
    parser = argparse.ArgumentParser(description='Запуск веб-приложения анализа повреждений автомобилей')
    parser.add_argument('--host', default='0.0.0.0', help='Хост для запуска (по умолчанию: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Порт для запуска (по умолчанию: 5000)')
    parser.add_argument('--debug', action='store_true', help='Запуск в режиме отладки')
    parser.add_argument('--install-deps', action='store_true', help='Установить зависимости перед запуском')
    parser.add_argument('--skip-checks', action='store_true', help='Пропустить проверки зависимостей и модели')
    
    args = parser.parse_args()
    
    print("🔍 Анализ повреждений автомобилей - Веб-приложение")
    print("=" * 50)
    
    # Установка зависимостей
    if args.install_deps:
        if not install_dependencies():
            sys.exit(1)
    
    # Проверки
    if not args.skip_checks:
        if not check_dependencies():
            print("\n💡 Для установки зависимостей запустите:")
            print("   python run_web_app.py --install-deps")
            sys.exit(1)
        
        if not check_model():
            print("\n💡 Для обучения модели запустите:")
            print("   python train_classifier.py")
            sys.exit(1)
    
    # Запуск приложения
    print("\n" + "=" * 50)
    run_app(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
