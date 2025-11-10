"""Weather data helpers (OpenWeather integration with caching)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import requests

from f1ml.config import get_project_paths


TRACK_COORDS: Dict[str, Tuple[float, float]] = {
    "australian-grand-prix": (-37.8497, 144.9680),
    "saudi-arabian-grand-prix": (21.6319, 39.1044),
    "bahrain-grand-prix": (26.0325, 50.5106),
    "miami-grand-prix": (25.9580, -80.2389),
    "monaco-grand-prix": (43.7340, 7.4200),
    "emilia-romagna-grand-prix": (44.3439, 11.7167),
    "spanish-grand-prix": (41.5689, 2.2578),
    "canadian-grand-prix": (45.5030, -73.5260),
    "hungarian-grand-prix": (47.5790, 19.2486),
    "italian-grand-prix": (45.6156, 9.2811),
    "s\u00e3o-paulo-grand-prix": (-23.7020, -46.6997),
}


def _weather_cache_path(event_slug: str, season: int, round_number: int) -> Path:
    weather_dir = get_project_paths().base_dir / "data" / "external" / "weather" / str(season)
    weather_dir.mkdir(parents=True, exist_ok=True)
    return weather_dir / f"r{round_number:02d}-{event_slug}.json"


def _default_weather_payload() -> Dict[str, float]:
    return {
        "temperature": 25.0,
        "rain_probability": 0.0,
        "wind_speed": 3.0,
        "humidity": 50.0,
        "source": "default",
    }


def _fetch_from_openweather(lat: float, lon: float, api_key: str) -> Dict[str, float]:
    url = "http://api.openweathermap.org/data/2.5/forecast"
    resp = requests.get(
        url,
        params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    forecasts = data.get("list") or []
    if not forecasts:
        return _default_weather_payload()
    sample = forecasts[0]
    return {
        "temperature": sample.get("main", {}).get("temp", 25.0),
        "rain_probability": sample.get("pop", 0.0),
        "wind_speed": sample.get("wind", {}).get("speed", 3.0),
        "humidity": sample.get("main", {}).get("humidity", 50.0),
        "timestamp": sample.get("dt_txt"),
        "source": "openweather",
    }


def get_weather_features(event_slug: str, season: int, round_number: int) -> Dict[str, float]:
    cache_path = _weather_cache_path(event_slug, season, round_number)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
    else:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        coords = TRACK_COORDS.get(event_slug)
        if api_key and coords:
            try:
                payload = _fetch_from_openweather(coords[0], coords[1], api_key)
                cache_path.write_text(json.dumps(payload, indent=2))
            except Exception:
                payload = _default_weather_payload()
        else:
            payload = _default_weather_payload()

    return {
        "weather_temp_c": payload.get("temperature", 25.0),
        "weather_rain_probability": payload.get("rain_probability", 0.0),
        "weather_wind_mps": payload.get("wind_speed", 3.0),
        "weather_humidity_pct": payload.get("humidity", 50.0),
    }
