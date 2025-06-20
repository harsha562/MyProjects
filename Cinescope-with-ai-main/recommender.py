# model/recommender.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import requests

class MovieRecommender:
    def __init__(self):
        self.model = None
        self.movies = None
        self.movie_features=None
        self.tmdb_api_key = "93914ec3e4107dfafe9d75e40500f70a"
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        
        self.tmdb_image_base_url = "https://image.tmdb.org/t/p/w500"

    def load_data(self, file_path):
        self.movies = pd.read_csv(file_path)
        print("Columns in dataset:", self.movies.columns.tolist())

    def train_model(self):
        if 'genre' not in self.movies.columns:
            raise ValueError("The dataset does not contain a 'genre' column.")
        
        # Assuming the dataset has a 'title' and 'genres' column
        self.movies['genre'] = self.movies['genre'].str.split('|')
        self.movies = self.movies.explode('genre')
        self.movie_features = pd.get_dummies(self.movies['genre'])
        self.model = NearestNeighbors(n_neighbors=5).fit(self.movie_features)

    def recommend(self, title):
        matching_movies = self.movies[self.movies['title'].str.lower() == title.lower()]
        if matching_movies.empty:
          return ["Sorry, we couldn't find that movie. Please try another title."], []
        movie_idx = self.movies[self.movies['title'] == title].index[0]
        movie_vector = self.movie_features.iloc[movie_idx].values.reshape(1, -1)  # Get the feature vector
        distances, indices = self.model.kneighbors(movie_vector)
        
        recommended_titles = self.movies.iloc[indices.flatten()]['title'].tolist()
        
        posters = []
        for rec_title in recommended_titles:
            poster_url = self.fetch_movie_poster(rec_title)
            posters.append(poster_url)
            
        print("Recommended Titles:",recommended_titles)
        print("Posters:",posters)
        
        return recommended_titles,posters

    def fetch_movie_poster(self, title):
        search_url = f"{self.tmdb_base_url}/search/movie"
        params = {
            'api_key': self.tmdb_api_key,
            'query': title
        }
        response = requests.get(search_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                poster_path = data['results'][0].get('poster_path')
                if poster_path:
                    return f"{self.tmdb_image_base_url}{poster_path}"
        return "Poster not available"
    