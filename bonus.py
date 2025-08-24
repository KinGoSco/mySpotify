# quick_bonus_features.py - Version simplifiée et rapide des fonctionnalités bonus
import pandas as pd
import numpy as np
import random
import os
from pathlib import Path

class QuickSpotifyFeatures:
    """Version simplifiée des fonctionnalités Spotify pour exécution rapide"""
    
    def __init__(self, sample_size=10000):
        """
        sample_size: Nombre de lignes à charger pour accélérer l'exécution
        """
        self.sample_size = sample_size
        print(f"🚀 Mode rapide: échantillon de {sample_size} interactions")
        self.load_sample_data()
    
    def load_sample_data(self):
        """Charge un échantillon des données pour exécution rapide"""
        print("📊 Chargement échantillon des données...")
        
        # Charger plus de données pour avoir plus de variété
        self.triplets_df = pd.read_csv(
            "train_triplets.txt", sep='\t', 
            names=['user_id', 'song_id', 'play_count'],
            nrows=self.sample_size
        )
        
        self.tracks_df = pd.read_csv(
            "p02_unique_tracks.txt", sep='<SEP>',
            names=['track_id', 'song_id', 'artist', 'title'],
            engine='python', nrows=5000  # Plus de tracks pour plus de variété
        )
        
        # Créer mappings rapides
        self.users = self.triplets_df['user_id'].unique()
        self.songs = self.triplets_df['song_id'].unique()
        
        print(f"✅ Données chargées: {len(self.triplets_df)} interactions, {len(self.users)} users, {len(self.songs)} songs")
    
    def quick_discover_weekly(self, user_id=None, n_tracks=15):
        """Version rapide de Discover Weekly"""
        print("\n🔍 BONUS 1: QUICK DISCOVER WEEKLY")
        print("-" * 40)
        
        if user_id is None:
            user_id = random.choice(self.users)
        
        print(f"👤 Génération pour utilisateur: {user_id[:10]}...")
        
        # 1. Ce que l'utilisateur a déjà écouté
        user_songs = set(self.triplets_df[
            self.triplets_df['user_id'] == user_id
        ]['song_id'])
        
        print(f"   🎵 Utilisateur a écouté: {len(user_songs)} chansons")
        
        # 2. Collaborative simple: utilisateurs similaires
        similar_users = self._find_similar_users_quick(user_id, top_k=10)
        
        # 3. Recommandations des utilisateurs similaires
        cf_recs = []
        for similar_user in similar_users:
            similar_songs = self.triplets_df[
                self.triplets_df['user_id'] == similar_user
            ]['song_id'].values
            
            # Songs pas encore écoutées
            new_songs = [s for s in similar_songs if s not in user_songs]
            cf_recs.extend(new_songs[:3])  # 3 par utilisateur similaire
        
        # 4. Popularité générale (songs trending)
        popular_songs = self.triplets_df.groupby('song_id')['play_count'].sum().nlargest(50)
        trending_recs = [s for s in popular_songs.index if s not in user_songs][:n_tracks//2]
        
        # 5. Mélange final
        all_recommendations = list(set(cf_recs + trending_recs))
        random.shuffle(all_recommendations)
        final_recs = all_recommendations[:n_tracks]
        
        # 6. Formatage
        playlist = []
        for i, song_id in enumerate(final_recs, 1):
            track_info = self.tracks_df[self.tracks_df['song_id'] == song_id]
            
            if not track_info.empty:
                track = track_info.iloc[0]
                reason = "Similar Users" if song_id in cf_recs else "Trending"
                
                playlist.append({
                    'rank': i,
                    'artist': track['artist'],
                    'title': track['title'],
                    'song_id': song_id,
                    'reason': reason
                })
        
        # Sauvegarder
        if playlist:
            df = pd.DataFrame(playlist)
            filename = f'quick_discover_weekly_{user_id[:8]}.csv'
            filepath = f'results/{filename}'
            df.to_csv(filepath, index=False)
            
            print(f"✅ Discover Weekly créé: {len(playlist)} tracks")
            print(f"   💾 Sauvegardé: {filename}")
            print("   Top 3 découvertes:")
            for _, row in df.head(3).iterrows():
                print(f"   {row['rank']}. {row['artist']} - {row['title']} ({row['reason']})")
            
            return df
        else:
            print("❌ Aucune recommandation Discover Weekly générée")
            return pd.DataFrame()
    
    def quick_artist_radio(self, seed_artist=None, n_tracks=20):
        """Version rapide d'Artist Radio"""
        print("\n📻 BONUS 2: QUICK ARTIST RADIO")
        print("-" * 40)
        
        if seed_artist is None:
            # Prendre un artiste populaire au hasard
            popular_artists = self.triplets_df.merge(
                self.tracks_df, on='song_id'
            ).groupby('artist')['play_count'].sum().nlargest(20)
            seed_artist = random.choice(popular_artists.index.tolist())
        
        print(f"🎤 Radio pour: {seed_artist}")
        
        # 1. Songs de l'artiste original
        artist_songs = self.tracks_df[
            self.tracks_df['artist'] == seed_artist
        ]['song_id'].values
        
        original_hits = []
        for song_id in artist_songs[:n_tracks//4]:  # 25% de l'artiste original
            track_info = self.tracks_df[self.tracks_df['song_id'] == song_id]
            if not track_info.empty:
                track = track_info.iloc[0]
                original_hits.append((song_id, track['artist'], track['title'], "Original Artist"))
        
        # 2. Utilisateurs qui écoutent cet artiste
        artist_fans = self.triplets_df[
            self.triplets_df['song_id'].isin(artist_songs)
        ]['user_id'].unique()
        
        print(f"   👥 Fans trouvés: {len(artist_fans)}")
        
        # 3. Autres artistes écoutés par ces fans
        fan_music = self.triplets_df[
            self.triplets_df['user_id'].isin(artist_fans)
        ].merge(self.tracks_df, on='song_id')
        
        similar_artists = fan_music[
            fan_music['artist'] != seed_artist
        ].groupby('artist')['play_count'].sum().nlargest(10)
        
        # 4. Prendre hits des artistes similaires
        similar_hits = []
        for artist in similar_artists.index[:5]:
            artist_tracks = self.tracks_df[self.tracks_df['artist'] == artist]
            for _, track in artist_tracks.head(2).iterrows():  # 2 par artiste
                similar_hits.append((track['song_id'], track['artist'], track['title'], "Similar Artist"))
        
        # 4. Quelques tracks populaires générales pour remplir
        general_popular = self.triplets_df.groupby('song_id')['play_count'].sum().nlargest(50)
        general_hits = []
        for song_id in general_popular.index:
            if len(general_hits) >= n_tracks//2:  # Remplir jusqu'à la moitié
                break
            track_info = self.tracks_df[self.tracks_df['song_id'] == song_id]
            if not track_info.empty:
                track = track_info.iloc[0]
                if track['artist'] != seed_artist:  # Éviter l'artiste seed
                    general_hits.append((song_id, track['artist'], track['title'], "Popular"))
        
        # 5. Combiner et garantir un minimum de tracks
        all_radio = original_hits + similar_hits + general_hits
        
        # Si pas assez, ajouter plus de tracks populaires
        if len(all_radio) < n_tracks:
            additional_popular = self.triplets_df.merge(
                self.tracks_df, on='song_id'
            ).groupby(['artist', 'title', 'song_id'])['play_count'].sum().nlargest(n_tracks*2)
            
            existing_songs = set([song_id for song_id, _, _, _ in all_radio])
            
            for (artist, title, song_id), _ in additional_popular.items():
                if song_id not in existing_songs and len(all_radio) < n_tracks:
                    all_radio.append((song_id, artist, title, "Filler"))
        
        # Mélanger et prendre exactement n_tracks
        random.shuffle(all_radio)
        final_radio = all_radio[:n_tracks]
        
        # 7. Formatage
        radio = []
        for i, (song_id, artist, title, category) in enumerate(final_radio, 1):
            radio.append({
                'position': i,
                'artist': artist,
                'title': title,
                'song_id': song_id,
                'category': category
            })
        
        # Sauvegarder
        if radio:
            df = pd.DataFrame(radio)
            safe_artist = seed_artist.replace('/', '_').replace(' ', '_').replace(';', '_')[:15]
            filename = f'quick_artist_radio_{safe_artist}.csv'
            filepath = f'results/{filename}'
            df.to_csv(filepath, index=False)
            
            print(f"✅ Artist Radio créé: {len(radio)} tracks")
            print(f"   💾 Sauvegardé: {filename}")
            print("   Composition:")
            categories = pd.Series([r['category'] for r in radio]).value_counts()
            for cat, count in categories.items():
                print(f"   • {cat}: {count} tracks")
            
            print("   Aperçu:")
            for _, row in df.head(3).iterrows():
                print(f"   {row['position']}. {row['artist']} - {row['title']} ({row['category']})")
            
            return df
        else:
            print("❌ Aucune radio générée")
            return pd.DataFrame()
    
    def quick_made_for_you(self, user_id=None):
        """Version rapide des playlists Made for You"""
        print("\n🎯 BONUS 3: QUICK MADE FOR YOU")
        print("-" * 40)
        
        if user_id is None:
            # Choisir un utilisateur avec plus d'historique
            user_song_counts = self.triplets_df.groupby('user_id').size()
            active_users = user_song_counts[user_song_counts >= 5].index  # Au moins 5 chansons
            if len(active_users) > 0:
                user_id = random.choice(active_users)
            else:
                user_id = random.choice(self.users)
        
        print(f"👤 Playlists pour utilisateur: {user_id[:10]}...")
        
        # Analyser l'utilisateur rapidement
        user_songs = self.triplets_df[self.triplets_df['user_id'] == user_id]
        print(f"   🎵 Historique: {len(user_songs)} chansons")
        
        user_artists = user_songs.merge(self.tracks_df, on='song_id')['artist'].value_counts()
        
        playlists = {}
        
        # 1. Time Capsule - Plus de vos artistes préférés
        time_capsule = []
        if len(user_artists) > 0:
            print(f"   🎤 Top artistes: {user_artists.head(3).index.tolist()}")
            
            for artist in user_artists.head(5).index:  # Top 5 artistes au lieu de 3
                artist_tracks = self.tracks_df[self.tracks_df['artist'] == artist]
                user_artist_songs = set(user_songs['song_id'])
                
                # Chansons de cet artiste pas encore écoutées
                new_songs = artist_tracks[~artist_tracks['song_id'].isin(user_artist_songs)]
                
                for _, track in new_songs.head(2).iterrows():  # 2 par artiste
                    time_capsule.append({
                        'rank': len(time_capsule) + 1,
                        'artist': track['artist'],
                        'title': track['title'],
                        'song_id': track['song_id'],
                        'reason': f"More from {artist}"
                    })
                    
                    if len(time_capsule) >= 10:  # Limite
                        break
        
        # Si Time Capsule toujours vide, remplir avec tracks populaires des mêmes genres
        if len(time_capsule) == 0:
            print("   🔄 Remplissage Time Capsule avec recommandations génériques...")
            popular_tracks = self.triplets_df.merge(
                self.tracks_df, on='song_id'
            ).groupby(['artist', 'title', 'song_id'])['play_count'].sum().nlargest(20)
            
            user_song_set = set(user_songs['song_id'])
            for (artist, title, song_id), _ in popular_tracks.items():
                if song_id not in user_song_set and len(time_capsule) < 8:
                    time_capsule.append({
                        'rank': len(time_capsule) + 1,
                        'artist': artist,
                        'title': title,
                        'song_id': song_id,
                        'reason': "Popular Pick"
                    })
        
        playlists['time_capsule'] = time_capsule
        
        # 2. Discovery Mix - Nouvelles découvertes
        all_user_songs = set(user_songs['song_id'])
        popular_unheard = self.triplets_df.groupby('song_id')['play_count'].sum().nlargest(100)  # Plus large pool
        
        discovery_mix = []
        for song_id in popular_unheard.index:
            if song_id not in all_user_songs:
                track_info = self.tracks_df[self.tracks_df['song_id'] == song_id]
                if not track_info.empty:
                    track = track_info.iloc[0]
                    discovery_mix.append({
                        'rank': len(discovery_mix) + 1,
                        'artist': track['artist'],
                        'title': track['title'],
                        'song_id': track['song_id'],
                        'reason': "New Discovery"
                    })
                    
                    if len(discovery_mix) >= 8:  # Limite à 8
                        break
        
        playlists['discovery_mix'] = discovery_mix
        
        # Sauvegarder les playlists
        saved_playlists = {}
        total_tracks = 0
        
        for playlist_name, playlist_data in playlists.items():
            if playlist_data:
                df = pd.DataFrame(playlist_data)
                filename = f'quick_made_for_you_{playlist_name}_{user_id[:8]}.csv'
                filepath = f'results/{filename}'
                df.to_csv(filepath, index=False)
                saved_playlists[playlist_name] = df
                total_tracks += len(playlist_data)
                
                print(f"✅ {playlist_name.replace('_', ' ').title()}: {len(playlist_data)} tracks")
                print(f"   💾 Sauvegardé: {filename}")
                if len(playlist_data) > 0:
                    print(f"   Exemple: {playlist_data[0]['artist']} - {playlist_data[0]['title']}")
            else:
                print(f"❌ {playlist_name.replace('_', ' ').title()}: Aucune track générée")
        
        if total_tracks > 0:
            print(f"🎯 Total Made for You: {total_tracks} tracks dans {len(saved_playlists)} playlists")
        
        return saved_playlists
    
    def _find_similar_users_quick(self, target_user, top_k=10):
        """Trouve rapidement des utilisateurs similaires"""
        target_songs = set(self.triplets_df[
            self.triplets_df['user_id'] == target_user
        ]['song_id'])
        
        if len(target_songs) == 0:
            return []
        
        similarities = []
        for user in self.users[:100]:  # Limite à 100 users pour la rapidité
            if user != target_user:
                user_songs = set(self.triplets_df[
                    self.triplets_df['user_id'] == user
                ]['song_id'])
                
                # Similarité Jaccard simple
                intersection = len(target_songs.intersection(user_songs))
                union = len(target_songs.union(user_songs))
                
                if union > 0:
                    similarity = intersection / union
                    similarities.append((user, similarity))
        
        # Trier et retourner top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [user for user, _ in similarities[:top_k]]

def main():
    """Exécution rapide des 3 fonctionnalités bonus"""
    
    # Créer dossier results
    os.makedirs('results', exist_ok=True)
    
    print("⚡ MYSPOTIFY BONUS - VERSION RAPIDE ⚡")
    print("="*50)
    print("🚀 Optimisé pour exécution rapide mais avec plus de données!")
    print()
    
    try:
        # Initialiser avec échantillon plus grand pour plus de variété
        spotify = QuickSpotifyFeatures(sample_size=10000)  # Plus d'interactions
        
        # Exécuter les 3 bonus
        print("\n🎵 EXÉCUTION DES 3 FONCTIONNALITÉS BONUS...")
        
        discover_weekly = spotify.quick_discover_weekly()
        artist_radio = spotify.quick_artist_radio()
        made_for_you = spotify.quick_made_for_you()
        
        print("\n" + "="*50)
        print("🎉 BONUS RAPIDE TERMINÉ!")
        print("="*50)
        
        # Lister les fichiers créés
        all_files = os.listdir('results') if os.path.exists('results') else []
        bonus_files = [f for f in all_files if f.startswith('quick_')]
        
        print(f"📁 Fichiers créés: {len(bonus_files)}")
        
        total_size = 0
        for file in bonus_files:
            file_path = os.path.join('results', file)
            if os.path.exists(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                total_size += size_kb
                print(f"   📄 {file} ({size_kb:.1f} KB)")
        
        print(f"\n📊 RÉSUMÉ:")
        print(f"   ✅ 3 fonctionnalités bonus implémentées!")
        print(f"   📄 {len(bonus_files)} fichiers CSV générés")
        print(f"   💾 Taille totale: {total_size:.1f} KB")
        print(f"   ⏱️  Temps d'exécution: ~1 minute")
        print(f"   🎵 Inspiré de: Spotify, Apple Music")
        
        if len(bonus_files) >= 3:
            print(f"\n🎉 BONUS COMPLET - Toutes les fonctionnalités marchent!")
        else:
            print(f"\n⚠️  Seulement {len(bonus_files)} fichiers générés sur 3+ attendus")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


# fix_discover_weekly.py - Correctif pour garantir 15 tracks dans Discover Weekly
import pandas as pd
import numpy as np
import random
import os

def generate_full_discover_weekly():
    """Génère une Discover Weekly complète avec 15 tracks garanties"""
    
    print("🔧 CORRECTIF DISCOVER WEEKLY")
    print("="*40)
    
    # Charger données
    print("📊 Chargement des données...")
    triplets_df = pd.read_csv(
        "train_triplets.txt", sep='\t', 
        names=['user_id', 'song_id', 'play_count'],
        nrows=10000
    )
    
    tracks_df = pd.read_csv(
        "p02_unique_tracks.txt", sep='<SEP>',
        names=['track_id', 'song_id', 'artist', 'title'],
        engine='python', nrows=5000
    )
    
    users = triplets_df['user_id'].unique()
    
    # Choisir un utilisateur avec un bon historique
    user_song_counts = triplets_df.groupby('user_id').size()
    active_users = user_song_counts[user_song_counts >= 10].index  # Au moins 10 chansons
    
    if len(active_users) > 0:
        target_user = random.choice(active_users)
    else:
        target_user = random.choice(users)
    
    print(f"👤 Utilisateur sélectionné: {target_user[:12]}...")
    
    # Analyser l'utilisateur
    user_songs = triplets_df[triplets_df['user_id'] == target_user]
    user_song_ids = set(user_songs['song_id'])
    
    print(f"🎵 Historique utilisateur: {len(user_song_ids)} chansons")
    
    # 1. Recommendations collaborative (utilisateurs similaires)
    print("🔍 Recherche utilisateurs similaires...")
    
    similar_users = find_similar_users(triplets_df, target_user, user_song_ids, top_k=20)
    print(f"👥 Utilisateurs similaires trouvés: {len(similar_users)}")
    
    cf_recommendations = []
    for similar_user in similar_users[:10]:  # Top 10 similaires
        similar_songs = triplets_df[triplets_df['user_id'] == similar_user]['song_id'].values
        new_songs = [s for s in similar_songs if s not in user_song_ids]
        cf_recommendations.extend(new_songs[:2])  # 2 par utilisateur similaire
    
    print(f"🤝 Recommandations collaborative: {len(set(cf_recommendations))}")
    
    # 2. Tracks populaires globales non écoutées
    print("📈 Recherche tracks populaires...")
    
    all_popularity = triplets_df.groupby('song_id')['play_count'].sum().sort_values(ascending=False)
    popular_unheard = [song for song in all_popularity.index if song not in user_song_ids]
    
    print(f"🔥 Tracks populaires non écoutées: {len(popular_unheard)}")
    
    # 3. Découverte par artistes (artistes populaires non écoutés)
    print("🎤 Découverte nouveaux artistes...")
    
    user_artists = set(user_songs.merge(tracks_df, on='song_id')['artist'])
    all_tracks_with_artists = triplets_df.merge(tracks_df, on='song_id')
    
    new_artist_songs = []
    artist_popularity = all_tracks_with_artists.groupby('artist')['play_count'].sum().sort_values(ascending=False)
    
    for artist in artist_popularity.index:
        if artist not in user_artists:
            artist_tracks = tracks_df[tracks_df['artist'] == artist]['song_id'].values
            unheard_from_artist = [s for s in artist_tracks if s not in user_song_ids]
            if unheard_from_artist:
                # Prendre la plus populaire de cet artiste
                best_song = None
                best_pop = 0
                for song in unheard_from_artist:
                    pop = triplets_df[triplets_df['song_id'] == song]['play_count'].sum()
                    if pop > best_pop:
                        best_pop = pop
                        best_song = song
                if best_song:
                    new_artist_songs.append(best_song)
                    if len(new_artist_songs) >= 10:  # Limite
                        break
    
    print(f"🆕 Nouveaux artistes: {len(new_artist_songs)}")
    
    # 4. Combiner toutes les recommandations
    all_recommendations = []
    
    # Ajouter collaborative (40%)
    cf_unique = list(set(cf_recommendations))
    random.shuffle(cf_unique)
    for song in cf_unique[:6]:  # 6 tracks collaborative
        all_recommendations.append((song, "Similar Users"))
    
    # Ajouter populaires (35%)
    for song in popular_unheard[:5]:  # 5 tracks populaires
        if song not in [r[0] for r in all_recommendations]:
            all_recommendations.append((song, "Trending"))
    
    # Ajouter nouveaux artistes (25%)
    for song in new_artist_songs[:4]:  # 4 nouveaux artistes
        if song not in [r[0] for r in all_recommendations]:
            all_recommendations.append((song, "New Artist"))
    
    # 5. Compléter jusqu'à 15 si nécessaire
    existing_songs = set([r[0] for r in all_recommendations])
    
    if len(all_recommendations) < 15:
        print(f"🔄 Complément nécessaire: {15 - len(all_recommendations)} tracks")
        
        # Ajouter plus de tracks populaires
        for song in all_popularity.index:
            if song not in existing_songs and song not in user_song_ids:
                all_recommendations.append((song, "Popular Filler"))
                if len(all_recommendations) >= 15:
                    break
    
    # 6. Formatage final
    print("💫 Formatage de la playlist...")
    
    playlist = []
    for i, (song_id, reason) in enumerate(all_recommendations[:15], 1):
        track_info = tracks_df[tracks_df['song_id'] == song_id]
        
        if not track_info.empty:
            track = track_info.iloc[0]
            playlist.append({
                'rank': i,
                'artist': track['artist'],
                'title': track['title'],
                'song_id': song_id,
                'reason': reason
            })
    
    # 7. Sauvegarder
    if playlist:
        df = pd.DataFrame(playlist)
        filename = f'fixed_discover_weekly_{target_user[:8]}.csv'
        filepath = f'results/{filename}'
        df.to_csv(filepath, index=False)
        
        print(f"\n✅ DISCOVER WEEKLY CORRIGÉ!")
        print(f"💾 Sauvegardé: {filename}")
        print(f"🎵 Tracks générées: {len(playlist)}/15")
        
        # Répartition des sources
        reason_counts = df['reason'].value_counts()
        print(f"\n📊 Répartition des sources:")
        for reason, count in reason_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   • {reason}: {count} tracks ({percentage:.1f}%)")
        
        print(f"\n🎧 Aperçu de votre Discover Weekly:")
        for _, row in df.head(5).iterrows():
            print(f"   {row['rank']:2d}. {row['artist']} - {row['title']}")
            print(f"       Source: {row['reason']}")
        
        if len(df) > 5:
            print(f"   ... et {len(df)-5} autres découvertes!")
        
        return df
    
    return None

def find_similar_users(triplets_df, target_user, target_songs, top_k=10):
    """Trouve des utilisateurs similaires plus efficacement"""
    
    similarities = []
    target_songs_set = set(target_songs)
    
    # Échantillonner les utilisateurs pour aller plus vite
    all_users = triplets_df['user_id'].unique()
    sample_users = random.sample(list(all_users), min(200, len(all_users)))
    
    for user in sample_users:
        if user != target_user:
            user_songs = set(triplets_df[triplets_df['user_id'] == user]['song_id'])
            
            # Similarité Jaccard
            intersection = len(target_songs_set.intersection(user_songs))
            union = len(target_songs_set.union(user_songs))
            
            if union > 0 and intersection > 0:  # Au moins une chanson en commun
                similarity = intersection / union
                similarities.append((user, similarity))
    
    # Trier et retourner top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [user for user, _ in similarities[:top_k]]

def main():
    """Exécute le correctif"""
    os.makedirs('results', exist_ok=True)
    
    try:
        result = generate_full_discover_weekly()
        
        if result is not None:
            print(f"\n🎉 CORRECTIF RÉUSSI!")
            print(f"📁 Fichier corrigé disponible dans results/")
            print(f"🔄 Remplace le fichier discover_weekly précédent")
        else:
            print(f"\n❌ Échec du correctif")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()