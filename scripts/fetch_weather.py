import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import subprocess

# Load environment variables
load_dotenv()

class EnvironmentalDataCollector:
    def __init__(self):
        # API Configuration
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        # Default location (New York City coordinates, can be parameterized)
        self.latitude = 40.7128
        self.longitude = -74.0060

    def fetch_weather_data(self):
        """
        Fetch current weather data from OpenWeatherMap
        """
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': self.latitude,
            'lon': self.longitude,
            'appid': self.api_key,
            'units': 'metric'  # Use 'imperial' for Fahrenheit
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract relevant weather metrics
            weather_data = {
                'timestamp': datetime.now().isoformat(),
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed']
            }

            return weather_data

        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None

    def fetch_air_quality_data(self):
        """
        Fetch air pollution data from OpenWeatherMap
        """
        base_url = "http://api.openweathermap.org/data/2.5/air_pollution"
        params = {
            'lat': self.latitude,
            'lon': self.longitude,
            'appid': self.api_key
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract relevant air quality metrics
            air_quality_data = {
                'timestamp': datetime.now().isoformat(),
                'aqi': data['list'][0]['main']['aqi'],
                'co': data['list'][0]['components']['co'],
                'no': data['list'][0]['components']['no'],
                'no2': data['list'][0]['components']['no2'],
                'o3': data['list'][0]['components']['o3'],
                'so2': data['list'][0]['components']['so2'],
                'pm2_5': data['list'][0]['components']['pm2_5'],
                'pm10': data['list'][0]['components']['pm10'],
                'nh3': data['list'][0]['components']['nh3']
            }

            return air_quality_data

        except requests.RequestException as e:
            print(f"Error fetching air quality data: {e}")
            return None

    def collect_and_save_data(self, output_file='D:/Work/MLops/course-project-Fai-zanAli/data/raw/environmental_data.csv'):
        """
        Collect data from multiple sources and append to CSV
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Fetch data
        weather_data = self.fetch_weather_data()
        air_quality_data = self.fetch_air_quality_data()

        # Merge data
        if weather_data and air_quality_data:
            combined_data = {**weather_data, **air_quality_data}

            # Create DataFrame
            df = pd.DataFrame([combined_data])

            # Append to file if it exists, else create a new file
            if os.path.exists(output_file):
                df.to_csv(output_file, mode='a', index=False, header=False)
            else:
                df.to_csv(output_file, mode='w', index=False, header=True)

            print(f"Data appended to {output_file}")

            # Automate DVC and Git commands
            #self.automate_dvc_git(output_file)

            return output_file

        print("Failed to fetch or save data.")
        return None

    def automate_dvc_git(self, file_path):
            """
            Automate DVC and Git commands to track and push new data
            """
            try:
                # Stage new data with DVC
                subprocess.run(['dvc', 'add', file_path], check=True)
                print(f"Staged {file_path} with DVC")

                # Commit changes to Git
                subprocess.run(['git', 'add', '.'], check=True)
                subprocess.run(['git', 'commit', '-m', 'Update environmental data file'], check=True)
                print("Git commit completed")

                # Push data to DVC remote storage
                subprocess.run(['dvc', 'push'], check=True)
                print("Data pushed to DVC remote storage")

                # Push changes to Git remote repository
                subprocess.run(['git', 'push', 'origin', 'main'], check=True)
                print("Changes pushed to Git remote repository")

            except subprocess.CalledProcessError as e:
                print(f"Error during DVC or Git process: {e}")

# Main execution
if __name__ == "__main__":
    collector = EnvironmentalDataCollector()
    collector.collect_and_save_data()
