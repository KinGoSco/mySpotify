# complete_myspotify.py - Système complet de recommandations
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import logging
from pathlib import Path
import pickle
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MySpotifyComplete:
    """Système complet de recommandation musicale MySpotify"""
    
    def __init__(self, data_path="."):
        self.data_path = Path(data_path)
        
        # Données
        self.triplets_df = None
        self.tracks_df = None
        self.lyrics_df = None
        self.word_mapping = None
        self.genre_df = None
        
        # Preprocessor
        self.user_to_idx = {}
        self.song_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_song = {}
        self.user_item_matrix = None
        
        # Résultats
        self.results = {}
    
    def load_all_data(self):
        """Charge toutes les données nécessaires"""
        logger.info("=== CHARGEMENT DE TOUTES LES DONNÉES ===")
        
        # Triplets
        logger.info("Chargement des triplets...")
        self.triplets_df = pd.read_csv(
            self.data_path / "train_triplets.txt", 
            sep='\t', names=['user_id', 'song_id', 'play_count']
        )
        logger.info(f"✅ Triplets: {len(self.triplets_df)} interactions")
        
        # Tracks
        logger.info("Chargement des tracks...")
        self.tracks_df = pd.read_csv(
            self.data_path / "p02_unique_tracks.txt", 
            sep='<SEP>', names=['track_id', 'song_id', 'artist', 'title'],
            engine='python'
        )
        logger.info(f"✅ Tracks: {len(self.tracks_df)} pistes")
        
        # Paroles
        logger.info("Chargement des paroles...")
        self.lyrics_df, self.word_mapping = self._load_lyrics()
        logger.info(f"✅ Paroles: {len(self.lyrics_df)} pistes avec paroles")
        
        # Genres
        logger.info("Chargement des genres...")
        self.genre_df = self._load_genres()
        logger.info(f"✅ Genres: {len(self.genre_df)} pistes avec genres")
        
        # Preprocessing
        self._create_mappings()
        self._build_user_item_matrix()
    
    def _load_lyrics(self):
        """Charge les données de paroles"""
        lyrics_data = []
        word_mapping = {}
        
        with open(self.data_path / "mxm_dataset_train.txt", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('#'):
                    continue
                
                if line.startswith('%'):
                    words = line[1:].split(',')
                    word_mapping = {i+1: word for i, word in enumerate(words)}
                    continue
                
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        track_id = parts[0]
                        mxm_track_id = parts[1]
                        
                        word_counts = {}
                        for part in parts[2:]:
                            if ':' in part:
                                try:
                                    word_idx, count = part.split(':')
                                    word_counts[int(word_idx)] = int(count)
                                except ValueError:
                                    continue
                        
                        lyrics_data.append({
                            'track_id': track_id,
                            'mxm_track_id': mxm_track_id,
                            'word_counts': word_counts
                        })
        
        return pd.DataFrame(lyrics_data), word_mapping
    
    def _load_genres(self):
        """Charge les données de genres"""
        genres_data = []
        with open(self.data_path / "p02_msd_tagtraum_cd2.cls", 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        track_id = parts[0]
                        majority_genre = parts[1]
                        minority_genre = parts[2] if len(parts) > 2 else None
                        
                        genres_data.append({
                            'track_id': track_id,
                            'majority_genre': majority_genre,
                            'minority_genre': minority_genre
                        })
        
        return pd.DataFrame(genres_data)
    
    def _create_mappings(self):
        """Crée les mappings bidirectionnels"""
        logger.info("Création des mappings...")
        
        unique_users = self.triplets_df['user_id'].unique()
        unique_songs = self.triplets_df['song_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.song_to_idx = {song: idx for idx, song in enumerate(unique_songs)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_song = {idx: song for song, idx in self.song_to_idx.items()}
        
        logger.info(f"✅ Mappings: {len(self.user_to_idx)} users, {len(self.song_to_idx)} songs")
    
    def _build_user_item_matrix(self):
        """Construit la matrice user-item"""
        logger.info("Construction de la matrice user-item...")
        
        user_indices = self.triplets_df['user_id'].map(self.user_to_idx)
        song_indices = self.triplets_df['song_id'].map(self.song_to_idx)
        play_counts = self.triplets_df['play_count']
        
        self.user_item_matrix = csr_matrix(
            (play_counts, (user_indices, song_indices)),
            shape=(len(self.user_to_idx), len(self.song_to_idx))
        )
        
        sparsity = 1 - self.user_item_matrix.nnz / np.prod(self.user_item_matrix.shape)
        logger.info(f"✅ Matrice: {self.user_item_matrix.shape}, sparsité: {sparsity:.6f}")
    
    def generate_top_250_tracks(self):
        """1. Top-250 tracks (Non-personnalisé)"""
        logger.info("=== 1. TOP-250 TRACKS ===")
        
        popularity = self.triplets_df.groupby('song_id')['play_count'].sum().sort_values(ascending=False)
        top_songs = popularity.head(250)
        
        results = []
        for i, (song_id, play_count) in enumerate(top_songs.items(), 1):
            track_info = self.tracks_df[self.tracks_df['song_id'] == song_id]
            if not track_info.empty:
                track_info = track_info.iloc[0]
                results.append({
                    'rank': i,
                    'artist': track_info['artist'],
                    'title': track_info['title'],
                    'song_id': song_id,
                    'play_count': int(play_count)
                })
        
        df_result = pd.DataFrame(results)
        df_result.to_csv('results/top_250_tracks.csv', index=False)
        self.results['top_250'] = df_result
        logger.info(f"✅ Top-250 généré: {len(df_result)} tracks")
        return df_result
    
    def generate_top_by_genre(self):
        """2. Top-100 par genre (Non-personnalisé)"""
        logger.info("=== 2. TOP-100 PAR GENRE ===")
        
        available_genres = self.genre_df['majority_genre'].unique()
        target_genres = ['Rock', 'Rap', 'Jazz', 'Electronic', 'Pop', 'Blues', 'Country', 'Reggae']
        
        all_genre_results = {}
        
        for genre in target_genres:
            if genre in available_genres:
                logger.info(f"Génération top-100 {genre}...")
                
                # Filtrer par genre
                genre_tracks = self.genre_df[
                    self.genre_df['majority_genre'] == genre
                ]['track_id'].values
                
                genre_songs = self.tracks_df[
                    self.tracks_df['track_id'].isin(genre_tracks)
                ]['song_id'].values
                
                genre_triplets = self.triplets_df[
                    self.triplets_df['song_id'].isin(genre_songs)
                ]
                
                if not genre_triplets.empty:
                    popularity = genre_triplets.groupby('song_id')['play_count'].sum().sort_values(ascending=False)
                    top_songs = popularity.head(100)
                    
                    results = []
                    for i, (song_id, play_count) in enumerate(top_songs.items(), 1):
                        track_info = self.tracks_df[self.tracks_df['song_id'] == song_id]
                        if not track_info.empty:
                            track_info = track_info.iloc[0]
                            results.append({
                                'rank': i,
                                'genre': genre,
                                'artist': track_info['artist'],
                                'title': track_info['title'],
                                'song_id': song_id,
                                'play_count': int(play_count)
                            })
                    
                    df_result = pd.DataFrame(results)
                    df_result.to_csv(f'results/top_100_{genre.lower()}.csv', index=False)
                    all_genre_results[genre] = df_result
                    logger.info(f"✅ {genre}: {len(df_result)} tracks")
        
        self.results['genres'] = all_genre_results
        return all_genre_results
    
    def generate_thematic_collections(self):
        """3. Collections thématiques (Content-based)"""
        logger.info("=== 3. COLLECTIONS THÉMATIQUES ===")
        
        themes = {
            'love': ['love', 'heart', 'kiss', 'romance', 'lover', 'baby', 'honey'],
            'war': ['war', 'fight', 'battle', 'soldier', 'gun', 'blood'],
            'happiness': ['happy', 'joy', 'smile', 'laugh', 'celebration', 'party'],
            'loneliness': ['lonely', 'alone', 'sad', 'cry', 'empty', 'miss'],
            'money': ['money', 'rich', 'dollar', 'gold', 'cash', 'wealth']
        }
        
        collections = {}
        
        for theme, keywords in themes.items():
            logger.info(f"Génération collection '{theme}'...")
            
            # Trouver les indices des mots-clés
            keyword_indices = []
            for keyword in keywords:
                for idx, word in self.word_mapping.items():
                    if keyword.lower() in word.lower():
                        keyword_indices.append(idx)
            
            if not keyword_indices:
                logger.warning(f"Aucun mot-clé trouvé pour {theme}")
                continue
            
            # Calculer score thématique pour chaque chanson
            theme_scores = []
            
            for _, row in self.lyrics_df.iterrows():
                track_id = row['track_id']
                word_counts = row['word_counts']
                
                # Score basé sur la présence des mots-clés
                theme_score = sum(word_counts.get(idx, 0) for idx in keyword_indices)
                
                if theme_score > 0:
                    # Trouver song_id et popularité
                    track_info = self.tracks_df[self.tracks_df['track_id'] == track_id]
                    if not track_info.empty:
                        song_id = track_info.iloc[0]['song_id']
                        song_popularity = self.triplets_df[
                            self.triplets_df['song_id'] == song_id
                        ]['play_count'].sum()
                        
                        if song_popularity > 0:
                            theme_scores.append({
                                'track_id': track_id,
                                'song_id': song_id,
                                'theme_score': theme_score,
                                'popularity': song_popularity,
                                'combined_score': theme_score * np.log(1 + song_popularity)
                            })
            
            # Trier et sélectionner top 50
            theme_scores.sort(key=lambda x: x['combined_score'], reverse=True)
            top_50 = theme_scores[:50]
            
            results = []
            for i, item in enumerate(top_50, 1):
                track_info = self.tracks_df[self.tracks_df['song_id'] == item['song_id']]
                if not track_info.empty:
                    track_info = track_info.iloc[0]
                    results.append({
                        'rank': i,
                        'theme': theme,
                        'artist': track_info['artist'],
                        'title': track_info['title'],
                        'song_id': item['song_id'],
                        'theme_score': item['theme_score'],
                        'play_count': int(item['popularity'])
                    })
            
            if results:
                df_result = pd.DataFrame(results)
                df_result.to_csv(f'results/collection_{theme}.csv', index=False)
                collections[theme] = df_result
                logger.info(f"✅ Collection '{theme}': {len(df_result)} tracks")
        
        self.results['collections'] = collections
        return collections
    
    def generate_user_based_recommendations(self, n_users=100):
        """4. People similar to you listen (User-based CF)"""
        logger.info("=== 4. USER-BASED COLLABORATIVE FILTERING ===")
        
        # Utiliser un sous-ensemble d'utilisateurs pour éviter les problèmes mémoire
        logger.info(f"Calcul pour les {n_users} premiers utilisateurs...")
        
        subset_matrix = self.user_item_matrix[:n_users, :]
        normalized_matrix = normalize(subset_matrix, norm='l2', axis=1)
        
        # Calculer similarité
        user_similarity = cosine_similarity(normalized_matrix)
        
        user_recommendations = []
        
        for user_idx in range(min(20, n_users)):  # Test sur 20 premiers utilisateurs
            user_id = self.idx_to_user[user_idx]
            
            # Trouver utilisateurs similaires
            similarities = user_similarity[user_idx]
            similarities[user_idx] = -1  # Exclure l'utilisateur lui-même
            
            similar_users = np.argsort(similarities)[::-1][:20]  # Top 20 similaires
            
            # Générer recommandations
            user_items = subset_matrix[user_idx].toarray()[0]
            recommendations = np.zeros(subset_matrix.shape[1])
            
            for similar_user_idx in similar_users:
                if similarities[similar_user_idx] > 0:
                    sim_score = similarities[similar_user_idx]
                    similar_items = subset_matrix[similar_user_idx].toarray()[0]
                    
                    unseen_items = (user_items == 0) & (similar_items > 0)
                    recommendations[unseen_items] += sim_score * similar_items[unseen_items]
            
            # Sélectionner top 10
            if recommendations.sum() > 0:
                top_items = np.argsort(recommendations)[::-1][:10]
                
                for rank, item_idx in enumerate(top_items, 1):
                    if recommendations[item_idx] > 0:
                        song_id = self.idx_to_song.get(item_idx, 'Unknown')
                        track_info = self.tracks_df[self.tracks_df['song_id'] == song_id]
                        
                        if not track_info.empty:
                            track_info = track_info.iloc[0]
                            user_recommendations.append({
                                'user_id': user_id,
                                'rank': rank,
                                'artist': track_info['artist'],
                                'title': track_info['title'],
                                'song_id': song_id,
                                'score': float(recommendations[item_idx])
                            })
        
        if user_recommendations:
            df_result = pd.DataFrame(user_recommendations)
            df_result.to_csv('results/user_based_recommendations.csv', index=False)
            self.results['user_cf'] = df_result
            logger.info(f"✅ Recommandations user-based: {len(df_result)} recs pour {df_result['user_id'].nunique()} users")
        
        return user_recommendations
    
    def generate_item_based_recommendations(self, n_items=100):
        """5. People who listen to this track usually listen (Item-based CF)"""
        logger.info("=== 5. ITEM-BASED COLLABORATIVE FILTERING ===")
        
        # Utiliser un sous-ensemble d'items
        subset_matrix = self.user_item_matrix[:, :n_items]
        
        # Transposer pour avoir items x users
        item_user_matrix = subset_matrix.T
        normalized_matrix = normalize(item_user_matrix, norm='l2', axis=1)
        
        # Calculer similarité entre items
        item_similarity = cosine_similarity(normalized_matrix)
        
        item_recommendations = []
        
        # Test sur quelques items populaires
        popular_items = np.array(subset_matrix.sum(axis=0)).flatten()
        top_items_idx = np.argsort(popular_items)[::-1][:20]  # 20 items les plus populaires
        
        for item_idx in top_items_idx:
            if item_idx < item_similarity.shape[0]:
                song_id = self.idx_to_song.get(item_idx, 'Unknown')
                
                # Trouver items similaires
                similarities = item_similarity[item_idx]
                similarities[item_idx] = -1  # Exclure l'item lui-même
                
                similar_items = np.argsort(similarities)[::-1][:10]
                
                for rank, similar_idx in enumerate(similar_items, 1):
                    if similarities[similar_idx] > 0.1:  # Seuil de similarité
                        similar_song_id = self.idx_to_song.get(similar_idx, 'Unknown')
                        track_info = self.tracks_df[self.tracks_df['song_id'] == similar_song_id]
                        
                        if not track_info.empty:
                            track_info = track_info.iloc[0]
                            item_recommendations.append({
                                'seed_song_id': song_id,
                                'rank': rank,
                                'artist': track_info['artist'],
                                'title': track_info['title'],
                                'recommended_song_id': similar_song_id,
                                'similarity': float(similarities[similar_idx])
                            })
        
        if item_recommendations:
            df_result = pd.DataFrame(item_recommendations)
            df_result.to_csv('results/item_based_recommendations.csv', index=False)
            self.results['item_cf'] = df_result
            logger.info(f"✅ Recommandations item-based: {len(df_result)} recs")
        
        return item_recommendations
    
    def run_complete_system(self):
        """Exécute le système complet de recommandations"""
        
        # Créer dossier results
        os.makedirs('results', exist_ok=True)
        
        # Charger toutes les données
        self.load_all_data()
        
        # Générer toutes les recommandations
        logger.info("\n🎵 GÉNÉRATION DE TOUTES LES RECOMMANDATIONS 🎵")
        
        self.generate_top_250_tracks()
        self.generate_top_by_genre()
        self.generate_thematic_collections()
        self.generate_user_based_recommendations()
        self.generate_item_based_recommendations()
        
        # Résumé final
        logger.info("=== RÉSUMÉ FINAL ===")
        logger.info("✅ Fichiers générés dans le dossier 'results/':")
        
        for file in Path('results').glob('*.csv'):
            size_kb = file.stat().st_size / 1024
            logger.info(f"   📄 {file.name} ({size_kb:.1f} KB)")
        
        logger.info("\n🎉 SYSTÈME MYSPOTIFY COMPLET EXÉCUTÉ AVEC SUCCÈS! 🎉")
        
        return self.results

def main():
    """Fonction principale"""
    try:
        myspotify = MySpotifyComplete()
        results = myspotify.run_complete_system()
        
        print("\n" + "="*50)
        print("🎵 MYSPOTIFY - SYSTÈME DE RECOMMANDATION MUSICAL 🎵")
        print("="*50)
        print(f"✅ Top-250 tracks générés")
        print(f"✅ {len(results.get('genres', {}))} genres traités")
        print(f"✅ {len(results.get('collections', {}))} collections thématiques")
        print(f"✅ Collaborative filtering utilisateur et item")
        print("\n📁 Tous les résultats sont dans le dossier 'results/'")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()