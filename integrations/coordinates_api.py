import requests
import os

# API ключи и базовые URL
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
WEATHER_API_URL = 'https://api.openweathermap.org/data/2.5/weather'
MAP_API_URL = 'https://example.com/migration-map-api'  # Замените на реальный URL API миграции

def get_weather_data(lat, lon):
    """
    Получение погодных данных по координатам.
    """
    params = {
        'lat': lat,
        'lon': lon,
        'appid': WEATHER_API_KEY,
        'units': 'metric'
    }
    response = requests.get(WEATHER_API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Ошибка получения данных о погоде: {response.status_code}, {response.text}")

def get_migration_data(lat, lon):
    """
    Сопоставление координат с картами миграции животных.
    """
    params = {
        'latitude': lat,
        'longitude': lon
    }
    response = requests.get(MAP_API_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Ошибка получения данных миграции: {response.status_code}, {response.text}")

def enrich_data_with_metadata(data):
    """
    Обогащение метаданными: добавление погоды и данных миграции.
    """
    for record in data:
        lat = record.get('latitude')
        lon = record.get('longitude')

        # Получение погодных данных
        try:
            weather_data = get_weather_data(lat, lon)
            record['weather'] = {
                'temperature': weather_data['main']['temp'],
                'humidity': weather_data['main']['humidity'],
                'conditions': weather_data['weather'][0]['description']
            }
        except Exception as e:
            print(f"Ошибка получения погоды для {lat}, {lon}: {e}")

        # Получение данных миграции
        try:
            migration_data = get_migration_data(lat, lon)
            record['migration'] = migration_data.get('migration_status', 'unknown')
        except Exception as e:
            print(f"Ошибка получения данных миграции для {lat}, {lon}: {e}")

    return data

