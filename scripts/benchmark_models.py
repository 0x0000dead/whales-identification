#!/usr/bin/env python3
"""
Скрипт для бенчмарка моделей API
Отправляет запросы к различным версиям моделей и собирает метрики производительности
"""

import asyncio
import aiohttp
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIBenchmark:
    def __init__(self, base_urls: Dict[str, str], test_images_dir: str):
        """
        Инициализация бенчмарка
        
        Args:
            base_urls: Словарь с именами моделей и их URL
            test_images_dir: Путь к директории с тестовыми изображениями
        """
        self.base_urls = base_urls
        self.test_images_dir = Path(test_images_dir)
        self.results = []
        
    async def test_single_request(self, session: aiohttp.ClientSession, 
                                 model_name: str, url: str, image_path: Path) -> Dict[str, Any]:
        """Тестирование одного запроса к API"""
        try:
            start_time = time.time()
            
            with open(image_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=image_path.name, 
                              content_type='image/jpeg')
                
                async with session.post(f"{url}/predict", data=data) as response:
                    end_time = time.time()
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        return {
                            'model_name': model_name,
                            'image_name': image_path.name,
                            'status': 'success',
                            'response_time': end_time - start_time,
                            'preprocessing_time': result.get('metrics', {}).get('preprocessing_time', 0),
                            'inference_time': result.get('metrics', {}).get('inference_time', 0),
                            'total_api_time': result.get('metrics', {}).get('total_time', 0),
                            'predictions': result.get('predictions', []),
                            'top1_confidence': result.get('predictions', [{}])[0].get('confidence', 0) if result.get('predictions') else 0,
                            'model_version': result.get('model_version', 'unknown'),
                            'file_size': image_path.stat().st_size
                        }
                    else:
                        return {
                            'model_name': model_name,
                            'image_name': image_path.name,
                            'status': 'error',
                            'response_time': end_time - start_time,
                            'error_code': response.status,
                            'error_message': await response.text()
                        }
                        
        except Exception as e:
            return {
                'model_name': model_name,
                'image_name': image_path.name,
                'status': 'exception',
                'error_message': str(e),
                'response_time': time.time() - start_time
            }

    async def run_benchmark(self, max_images: int = 50, concurrent_requests: int = 5) -> pd.DataFrame:
        """Запуск бенчмарка для всех моделей"""
        
        # Получаем список тестовых изображений
        image_files = list(self.test_images_dir.glob("*.jpg"))[:max_images]
        if not image_files:
            raise ValueError(f"Не найдено изображений в {self.test_images_dir}")
        
        logger.info(f"Запуск бенчмарка для {len(image_files)} изображений и {len(self.base_urls)} моделей")
        
        # Проверка доступности API
        await self._check_apis_health()
        
        # Семафор для ограничения количества одновременных запросов
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            tasks = []
            
            for model_name, url in self.base_urls.items():
                for image_path in image_files:
                    task = self._limited_request(semaphore, session, model_name, url, image_path)
                    tasks.append(task)
            
            logger.info(f"Выполнение {len(tasks)} запросов...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Фильтруем исключения
            valid_results = [r for r in results if isinstance(r, dict)]
            self.results = valid_results
            
        return pd.DataFrame(valid_results)

    async def _limited_request(self, semaphore: asyncio.Semaphore, 
                              session: aiohttp.ClientSession, 
                              model_name: str, url: str, image_path: Path) -> Dict[str, Any]:
        """Ограниченный запрос с семафором"""
        async with semaphore:
            return await self.test_single_request(session, model_name, url, image_path)

    async def _check_apis_health(self):
        """Проверка доступности всех API"""
        async with aiohttp.ClientSession() as session:
            for model_name, url in self.base_urls.items():
                try:
                    async with session.get(f"{url}/health") as response:
                        if response.status != 200:
                            logger.warning(f"API {model_name} ({url}) недоступно: {response.status}")
                        else:
                            logger.info(f"API {model_name} ({url}) работает")
                except Exception as e:
                    logger.error(f"Ошибка подключения к {model_name} ({url}): {e}")

    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ результатов бенчмарка"""
        
        successful_df = df[df['status'] == 'success'].copy()
        
        if successful_df.empty:
            return {"error": "Нет успешных запросов для анализа"}
        
        analysis = {}
        
        # Общая статистика по моделям
        model_stats = successful_df.groupby('model_name').agg({
            'response_time': ['mean', 'std', 'min', 'max', 'count'],
            'inference_time': ['mean', 'std'],
            'preprocessing_time': ['mean', 'std'],
            'top1_confidence': ['mean', 'std'],
            'file_size': 'mean'
        }).round(4)
        
        analysis['model_performance'] = model_stats.to_dict()
        
        # Статистика ошибок
        error_stats = df.groupby('model_name')['status'].value_counts().unstack(fill_value=0)
        if 'success' in error_stats.columns:
            error_stats['success_rate'] = error_stats['success'] / error_stats.sum(axis=1)
        analysis['error_statistics'] = error_stats.to_dict()
        
        # Сравнение производительности
        perf_comparison = successful_df.groupby('model_name').agg({
            'response_time': 'mean',
            'inference_time': 'mean',
            'top1_confidence': 'mean'
        }).round(4)
        
        # Ранжирование моделей
        perf_comparison['speed_rank'] = perf_comparison['response_time'].rank()
        perf_comparison['confidence_rank'] = perf_comparison['top1_confidence'].rank(ascending=False)
        perf_comparison['overall_score'] = (
            (1 / perf_comparison['speed_rank']) * 0.4 + 
            (1 / perf_comparison['confidence_rank']) * 0.6
        )
        
        analysis['performance_comparison'] = perf_comparison.to_dict()
        
        # Рекомендации
        best_speed = perf_comparison['response_time'].idxmin()
        best_accuracy = perf_comparison['top1_confidence'].idxmax()
        best_overall = perf_comparison['overall_score'].idxmax()
        
        analysis['recommendations'] = {
            'fastest_model': best_speed,
            'most_accurate_model': best_accuracy,
            'best_overall_model': best_overall,
            'speed_difference': f"{perf_comparison.loc[best_accuracy, 'response_time'] / perf_comparison.loc[best_speed, 'response_time']:.2f}x",
            'accuracy_difference': f"{perf_comparison.loc[best_accuracy, 'top1_confidence'] / perf_comparison.loc[best_speed, 'top1_confidence']:.2f}x"
        }
        
        return analysis

    def create_visualizations(self, df: pd.DataFrame, output_dir: str = "benchmark_results"):
        """Создание визуализаций результатов"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        successful_df = df[df['status'] == 'success'].copy()
        
        if successful_df.empty:
            logger.warning("Нет данных для визуализации")
            return
        
        # Настройка стиля
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                pass  # Используем стиль по умолчанию
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Время отклика по моделям
        sns.boxplot(data=successful_df, x='model_name', y='response_time', ax=axes[0,0])
        axes[0,0].set_title('Время отклика по моделям')
        axes[0,0].set_xlabel('Модель')
        axes[0,0].set_ylabel('Время отклика (сек)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Уверенность предсказаний
        sns.boxplot(data=successful_df, x='model_name', y='top1_confidence', ax=axes[0,1])
        axes[0,1].set_title('Уверенность предсказаний')
        axes[0,1].set_xlabel('Модель')
        axes[0,1].set_ylabel('Уверенность (Top-1)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Время инференса vs preprocessing
        model_times = successful_df.groupby('model_name')[['preprocessing_time', 'inference_time']].mean()
        model_times.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Время обработки по этапам')
        axes[1,0].set_xlabel('Модель')
        axes[1,0].set_ylabel('Время (сек)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend()
        
        # 4. Scatter plot: скорость vs точность
        model_avg = successful_df.groupby('model_name').agg({
            'response_time': 'mean',
            'top1_confidence': 'mean'
        })
        
        axes[1,1].scatter(model_avg['response_time'], model_avg['top1_confidence'], s=100)
        for model_name, row in model_avg.iterrows():
            axes[1,1].annotate(model_name, (row['response_time'], row['top1_confidence']), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1,1].set_title('Скорость vs Точность')
        axes[1,1].set_xlabel('Время отклика (сек)')
        axes[1,1].set_ylabel('Уверенность (Top-1)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'benchmark_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Создание отдельного графика успешности запросов
        fig, ax = plt.subplots(figsize=(10, 6))
        error_stats = df.groupby('model_name')['status'].value_counts().unstack(fill_value=0)
        if 'success' in error_stats.columns:
            success_rate = error_stats['success'] / error_stats.sum(axis=1)
            success_rate.plot(kind='bar', ax=ax)
            ax.set_title('Процент успешных запросов по моделям')
            ax.set_xlabel('Модель')
            ax.set_ylabel('Процент успешных запросов')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path / 'success_rate.png', dpi=300, bbox_inches='tight')
            plt.close()

    def save_results(self, df: pd.DataFrame, analysis: Dict[str, Any], output_dir: str = "benchmark_results"):
        """Сохранение результатов"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Сохранение сырых данных
        df.to_csv(output_path / 'raw_results.csv', index=False)
        
        # Сохранение анализа
        with open(output_path / 'analysis.json', 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        # Создание отчета в текстовом формате
        self._create_text_report(df, analysis, output_path / 'report.txt')
        
        logger.info(f"Результаты сохранены в {output_path}")

    def _create_text_report(self, df: pd.DataFrame, analysis: Dict[str, Any], output_file: Path):
        """Создание текстового отчета"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ПО БЕНЧМАРКУ МОДЕЛЕЙ API\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Дата проведения: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Общее количество запросов: {len(df)}\n")
            f.write(f"Успешных запросов: {len(df[df['status'] == 'success'])}\n")
            f.write(f"Неудачных запросов: {len(df[df['status'] != 'success'])}\n\n")
            
            if 'recommendations' in analysis:
                f.write("РЕКОМЕНДАЦИИ\n")
                f.write("-" * 20 + "\n")
                rec = analysis['recommendations']
                f.write(f"Самая быстрая модель: {rec['fastest_model']}\n")
                f.write(f"Самая точная модель: {rec['most_accurate_model']}\n")
                f.write(f"Лучшая общая модель: {rec['best_overall_model']}\n")
                f.write(f"Разница в скорости: {rec['speed_difference']}\n")
                f.write(f"Разница в точности: {rec['accuracy_difference']}\n\n")
            
            if 'performance_comparison' in analysis:
                f.write("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ\n")
                f.write("-" * 30 + "\n")
                perf = analysis['performance_comparison']
                for model in perf['response_time']:
                    f.write(f"\n{model}:\n")
                    f.write(f"  Среднее время отклика: {perf['response_time'][model]:.4f} сек\n")
                    f.write(f"  Среднее время инференса: {perf['inference_time'][model]:.4f} сек\n")
                    f.write(f"  Средняя уверенность: {perf['top1_confidence'][model]:.4f}\n")
                    f.write(f"  Общий рейтинг: {perf['overall_score'][model]:.4f}\n")

async def main():
    parser = argparse.ArgumentParser(description='Бенчмарк API моделей')
    parser.add_argument('--images-dir', default='./data/datasets', 
                       help='Путь к директории с тестовыми изображениями')
    parser.add_argument('--max-images', type=int, default=20,
                       help='Максимальное количество изображений для тестирования')
    parser.add_argument('--concurrent', type=int, default=3,
                       help='Количество одновременных запросов')
    parser.add_argument('--output-dir', default='./benchmark_results',
                       help='Директория для сохранения результатов')
    
    args = parser.parse_args()
    
    # URL API для разных моделей
    api_urls = {
        'efficientnet_v1': 'http://localhost:8001',
        'efficientnet_v2': 'http://localhost:8002', 
        'resnet_v1': 'http://localhost:8003'
    }
    
    # Создание бенчмарка
    benchmark = APIBenchmark(api_urls, args.images_dir)
    
    # Запуск бенчмарка
    logger.info("Запуск бенчмарка...")
    df = await benchmark.run_benchmark(
        max_images=args.max_images, 
        concurrent_requests=args.concurrent
    )
    
    # Анализ результатов
    logger.info("Анализ результатов...")
    analysis = benchmark.analyze_results(df)
    
    # Создание визуализаций
    logger.info("Создание визуализаций...")
    benchmark.create_visualizations(df, args.output_dir)
    
    # Сохранение результатов
    logger.info("Сохранение результатов...")
    benchmark.save_results(df, analysis, args.output_dir)
    
    # Вывод краткого отчета
    if 'recommendations' in analysis:
        rec = analysis['recommendations']
        print("\n" + "="*50)
        print("КРАТКИЙ ОТЧЕТ")
        print("="*50)
        print(f"Самая быстрая модель: {rec['fastest_model']}")
        print(f"Самая точная модель: {rec['most_accurate_model']}")
        print(f"Лучшая общая модель: {rec['best_overall_model']}")
        print(f"Результаты сохранены в: {args.output_dir}")

if __name__ == "__main__":
    asyncio.run(main()) 