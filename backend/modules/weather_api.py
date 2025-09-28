"""
JARVIS Mark 5 - Weather API Integration
Provides real-time weather information
"""

import os
import json
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class WeatherAPI:
    """Weather API integration for JARVIS"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = self._load_api_key()
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    def _load_api_key(self) -> Optional[str]:
        """Load weather API key from config"""
        try:
            with open('config/config.json') as f:
                config = json.load(f)
                return config.get('WEATHER_API_KEY')
        except Exception as e:
            self.logger.warning(f"Could not load weather API key: {e}")
            return None
    
    def get_current_weather(self, city: str = "London") -> Dict[str, Any]:
        """Get current weather for a city"""
        try:
            if not self.api_key:
                return self._get_demo_weather(city)
            
            url = f"{self.base_url}/weather"
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                "success": True,
                "city": data['name'],
                "country": data['sys']['country'],
                "temperature": f"{data['main']['temp']:.1f}°C",
                "feels_like": f"{data['main']['feels_like']:.1f}°C",
                "condition": data['weather'][0]['description'].title(),
                "humidity": f"{data['main']['humidity']}%",
                "wind_speed": f"{data['wind']['speed']} m/s",
                "pressure": f"{data['main']['pressure']} hPa",
                "visibility": f"{data.get('visibility', 0) / 1000:.1f} km"
            }
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Weather API request failed: {e}")
            return self._get_demo_weather(city)
        except Exception as e:
            self.logger.error(f"Weather API error: {e}")
            return self._get_demo_weather(city)
    
    def _get_demo_weather(self, city: str) -> Dict[str, Any]:
        """Get demo weather data when API is not available"""
        import random
        
        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Snowy"]
        condition = random.choice(conditions)
        
        # Generate realistic temperature based on condition
        if condition == "Sunny":
            temp = random.uniform(20, 30)
        elif condition == "Partly Cloudy":
            temp = random.uniform(15, 25)
        elif condition == "Cloudy":
            temp = random.uniform(10, 20)
        elif condition == "Rainy":
            temp = random.uniform(5, 15)
        else:  # Snowy
            temp = random.uniform(-5, 5)
        
        return {
            "success": True,
            "city": city,
            "country": "Demo",
            "temperature": f"{temp:.1f}°C",
            "feels_like": f"{temp + random.uniform(-2, 2):.1f}°C",
            "condition": condition,
            "humidity": f"{random.randint(40, 80)}%",
            "wind_speed": f"{random.uniform(2, 15):.1f} m/s",
            "pressure": f"{random.randint(1000, 1020)} hPa",
            "visibility": f"{random.uniform(5, 15):.1f} km",
            "note": "Demo data - API key not configured"
        }
    
    def get_weather_forecast(self, city: str = "London", days: int = 5) -> Dict[str, Any]:
        """Get weather forecast for multiple days"""
        try:
            if not self.api_key:
                return self._get_demo_forecast(city, days)
            
            url = f"{self.base_url}/forecast"
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 8 forecasts per day (every 3 hours)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Process forecast data
            forecasts = []
            for item in data['list'][:days * 8:8]:  # Get one forecast per day
                forecasts.append({
                    "date": item['dt_txt'].split()[0],
                    "time": item['dt_txt'].split()[1],
                    "temperature": f"{item['main']['temp']:.1f}°C",
                    "condition": item['weather'][0]['description'].title(),
                    "humidity": f"{item['main']['humidity']}%",
                    "wind_speed": f"{item['wind']['speed']} m/s"
                })
            
            return {
                "success": True,
                "city": data['city']['name'],
                "country": data['city']['country'],
                "forecasts": forecasts
            }
            
        except Exception as e:
            self.logger.error(f"Weather forecast error: {e}")
            return self._get_demo_forecast(city, days)
    
    def _get_demo_forecast(self, city: str, days: int) -> Dict[str, Any]:
        """Get demo forecast data"""
        import random
        from datetime import datetime, timedelta
        
        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Snowy"]
        forecasts = []
        
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            condition = random.choice(conditions)
            temp = random.uniform(10, 25)
            
            forecasts.append({
                "date": date,
                "time": "12:00:00",
                "temperature": f"{temp:.1f}°C",
                "condition": condition,
                "humidity": f"{random.randint(40, 80)}%",
                "wind_speed": f"{random.uniform(2, 15):.1f} m/s"
            })
        
        return {
            "success": True,
            "city": city,
            "country": "Demo",
            "forecasts": forecasts,
            "note": "Demo data - API key not configured"
        }
    
    def format_weather_response(self, weather_data: Dict[str, Any]) -> str:
        """Format weather data into a readable response"""
        if not weather_data.get("success"):
            return "Sorry, I couldn't get the weather information."
        
        response = f"The current weather in {weather_data['city']}, {weather_data['country']} is "
        response += f"{weather_data['temperature']} and {weather_data['condition']}. "
        response += f"It feels like {weather_data['feels_like']}. "
        response += f"Humidity is {weather_data['humidity']} and wind speed is {weather_data['wind_speed']}. "
        response += f"Pressure is {weather_data['pressure']} and visibility is {weather_data['visibility']}."
        
        if weather_data.get("note"):
            response += f" Note: {weather_data['note']}"
        
        return response

# Example usage
if __name__ == "__main__":
    weather = WeatherAPI()
    
    # Test current weather
    print("Current Weather Test:")
    current = weather.get_current_weather("London")
    print(weather.format_weather_response(current))
    
    print("\n" + "="*50 + "\n")
    
    # Test forecast
    print("Weather Forecast Test:")
    forecast = weather.get_weather_forecast("London", 3)
    if forecast.get("success"):
        print(f"Weather forecast for {forecast['city']}, {forecast['country']}:")
        for day in forecast['forecasts']:
            print(f"  {day['date']}: {day['temperature']}, {day['condition']}")
