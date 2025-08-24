# main.py - Script principal pour tester MySpotify
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Classe pour charger tous les datasets"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
    
    def load_triplets(self, filename="train_triplet.txt"):
        """Charge les triplets (user_id, song_id, play_count)"""
        logger.info(f"Chargement des triplets depuis {filename}...")
        filepath = self.data_path / filename
        
        df = pd.read_csv(filepath, sep='\t', 
                        names=['user_id', 'song_id', 'play_count'])
        logger.info(f"Triplets chargés: {len(df)} interactions")
        return df
    
    def load_tracks(self, filename="p02_unique_tracks.txt"):
        """Charge les informations des tracks"""
        logger.info(f"Chargement des tracks depuis {filename}...")
        filepath = self.data_path / filename
        
        df = pd.read_csv(filepath, sep='<SEP>', 
                        names=['track_id', 'song_id', 'artist', 'title'],
                        engine='python')
        logger.info(f"Tracks chargés: {len(df)} pistes")
        return df
    
    def load_lyrics(self, filename="mxm_dataset_train.txt"):
        """Charge les données de paroles"""
        logger.info(f"Chargement des paroles depuis {filename}...")
        filepath = self.data_path / filename
        
        lyrics_data = []
        word_mapping = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments
                if line.startswith('#'):
                    continue
                
                # Get word mapping
                if line.startswith('%'):
                    words = line[1:].split(',')
                    word_mapping = {i+1: word for i, word in enumerate(words)}
                    continue
                
                # Parse lyrics data
                if line:
                    parts = line.split(',')
                    track_id = parts[0]
                    mxm_track_id = parts[1]
                    
                    word_counts = {}
                    for part in parts[2:]:
                        if ':' in part:
                            word_idx, count = part.split(':')
                            word_counts[int(word_idx)] = int(count)
                    
                    lyrics_data.append({
                        'track_id': track_id,
                        'mxm_track_id': mxm_track_id,
                        'word_counts': word_counts
                    })
        
        df = pd.DataFrame(lyrics_data)
        logger.info(f"Paroles chargées: {len(df)} pistes avec paroles")
        return df, word_mapping
    
    def load_genres(self, filename="p02_msd_tagtraum_cd2.cls.txt"):
        """Charge les genres"""
        logger.info(f"Chargement des genres depuis {filename}...")
        filepath = self.data_path / filename
        
        genres_data = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('\t')
                    track_id = parts[0]
                    majority_genre = parts[1]
                    minority_genre = parts[2] if len(parts) > 2 else None
                    
                    genres_data.append({
                        'track_id': track_id,
                        'majority_genre': majority_genre,
                        'minority_genre': minority_genre
                    })
        
        df = pd.DataFrame(genres_data)
        logger.info(f"Genres chargés: {len(df)} pistes avec genres")
        return df

class DataPreprocessor:
    """Classe pour préprocesser les données"""
    
    def __init__(self):
        self.user_to_idx = {}
        self.song_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_song = {}
    
    def create_mappings(self, triplets_df):
        """Crée les mappings bidirectionnels"""
        logger.info("Création des mappings user/song...")
        
        unique_users = triplets_df['user_id'].unique()
        unique_songs = triplets_df['song_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.song_to_idx = {song: idx for idx, song in enumerate(unique_songs)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_song = {idx: song for song, idx in self.song_to_idx.items()}
        
        logger.info(f"Mappings créés: {len(self.user_to_idx)} users, {len(self.song_to_idx)} songs")
    
    def build_user_item_matrix(self, triplets_df):
        """Construit la matrice user-item sparse"""
        logger.info("Construction de la matrice user-item...")
        
        user_indices = triplets_df['user_id'].map(self.user_to_idx)
        song_indices = triplets_df['song_id'].map(self.song_to_idx)
        play_counts = triplets_df['play_count']
        
        matrix = csr_matrix(
            (play_counts, (user_indices, song_indices)),
            shape=(len(self.user_to_idx), len(self.song_to_idx))
        )
        
        logger.info(f"Matrice construite: {matrix.shape}, sparsité: {1 - matrix.nnz / np.prod(matrix.shape):.4f}")
        return matrix

class PopularityRecommender:
    """Recommandeur basé sur la popularité"""
    
    def __init__(self, tracks_df, triplets_df):
        self.tracks_df = tracks_df
        self.triplets_df = triplets_df
    
    def get_top_tracks(self, n=250):
        """Retourne les n tracks les plus populaires"""
        logger.info(f"Génération du top-{n} tracks...")
        
        # Calculer la popularité globale
        popularity = self.triplets_df.groupby('song_id')['play_count'].sum().sort_values(ascending=False)
        top_songs = popularity.head(n)
        
        results = []
        for i, (song_id, play_count) in enumerate(top_songs.items(), 1):
            # Trouver les infos du track
            track_info = self.tracks_df[self.tracks_df['song_id'] == song_id]
            if not track_info.empty:
                track_info = track_info.iloc[0]
                results.append({
                    'index': i,
                    'artist': track_info['artist'],
                    'title': track_info['title'],
                    'song_id': song_id,
                    'play_count': int(play_count)
                })
        
        df_result = pd.DataFrame(results)
        logger.info(f"Top-{n} généré avec {len(df_result)} tracks")
        return df_result

class GenreRecommender:
    """Recommandeur basé sur les genres"""
    
    def __init__(self, tracks_df, triplets_df, genre_df):
        self.tracks_df = tracks_df
        self.triplets_df = triplets_df
        self.genre_df = genre_df
    
    def get_top_by_genre(self, genre, n=100):
        """Retourne les n tracks les plus populaires pour un genre"""
        logger.info(f"Génération du top-{n} pour le genre {genre}...")
        
        # Filtrer par genre
        genre_tracks = self.genre_df[
            self.genre_df['majority_genre'] == genre
        ]['track_id'].values
        
        # Mapper track_id vers song_id
        genre_songs = self.tracks_df[
            self.tracks_df['track_id'].isin(genre_tracks)
        ]['song_id'].values
        
        # Calculer popularité dans ce genre
        genre_triplets = self.triplets_df[
            self.triplets_df['song_id'].isin(genre_songs)
        ]
        
        if genre_triplets.empty:
            logger.warning(f"Aucune chanson trouvée pour le genre {genre}")
            return pd.DataFrame()
        
        popularity = genre_triplets.groupby('song_id')['play_count'].sum().sort_values(ascending=False)
        top_songs = popularity.head(n)
        
        results = []
        for i, (song_id, play_count) in enumerate(top_songs.items(), 1):
            track_info = self.tracks_df[self.tracks_df['song_id'] == song_id]
            if not track_info.empty:
                track_info = track_info.iloc[0]
                results.append({
                    'index': i,
                    'genre': genre,
                    'artist': track_info['artist'],
                    'title': track_info['title'],
                    'song_id': song_id,
                    'play_count': int(play_count)
                })
        
        df_result = pd.DataFrame(results)
        logger.info(f"Top-{n} {genre} généré avec {len(df_result)} tracks")
        return df_result

def main():
    """Fonction principale pour tester le système"""
    
    # Configuration
    DATA_PATH = "."  # Ajustez selon votre structure de dossiers
    
    try:
        # 1. Chargement des données
        logger.info("=== PHASE 1: CHARGEMENT DES DONNÉES ===")
        data_loader = DataLoader(DATA_PATH)
        
        triplets_df = data_loader.load_triplets()
        tracks_df = data_loader.load_tracks()
        lyrics_df, word_mapping = data_loader.load_lyrics()
        genre_df = data_loader.load_genres()
        
        # 2. Preprocessing
        logger.info("=== PHASE 2: PREPROCESSING ===")
        preprocessor = DataPreprocessor()
        preprocessor.create_mappings(triplets_df)
        user_item_matrix = preprocessor.build_user_item_matrix(triplets_df)
        
        # 3. Test des recommandeurs non-personnalisés
        logger.info("=== PHASE 3: RECOMMANDATIONS NON-PERSONNALISÉES ===")
        
        # Top-250 tracks
        popularity_rec = PopularityRecommender(tracks_df, triplets_df)
        top_250 = popularity_rec.get_top_tracks(250)
        print("Top 10 des tracks les plus populaires:")
        print(top_250.head(10)[['index', 'artist', 'title', 'play_count']])
        print()
        
        # Top-100 par genre
        genre_rec = GenreRecommender(tracks_df, triplets_df, genre_df)
        available_genres = genre_df['majority_genre'].unique()
        print(f"Genres disponibles: {available_genres}")
        
        # Test avec le genre "Rock"
        if 'Rock' in available_genres:
            top_rock = genre_rec.get_top_by_genre('Rock', 10)
            print("Top 10 Rock:")
            print(top_rock[['index', 'artist', 'title', 'play_count']])
        
        # 4. Statistiques générales
        logger.info("=== STATISTIQUES GÉNÉRALES ===")
        print(f"Nombre total d'utilisateurs: {len(preprocessor.user_to_idx)}")
        print(f"Nombre total de chansons: {len(preprocessor.song_to_idx)}")
        print(f"Nombre total d'interactions: {len(triplets_df)}")
        print(f"Sparsité de la matrice: {1 - user_item_matrix.nnz / np.prod(user_item_matrix.shape):.6f}")
        
        # 5. Sauvegarde des résultats
        logger.info("=== SAUVEGARDE ===")
        top_250.to_csv('top_250_tracks.csv', index=False)
        logger.info("Résultats sauvegardés dans top_250_tracks.csv")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        raise

if __name__ == "__main__":
    main()