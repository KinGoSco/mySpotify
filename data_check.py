# data_check.py - Script pour vérifier la qualité des données
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_data_files():
    """Vérifie que tous les fichiers de données sont présents et lisibles"""
    
    data_files = {
        'train_triplet.txt': 'Triplets (user_id, song_id, play_count)',
        'p02_unique_tracks.txt': 'Informations des tracks',
        'mxm_dataset_train.txt': 'Données de paroles',
        'p02_msd_tagtraum_cd2.cls.txt': 'Genres musicaux'
    }
    
    print("=== VÉRIFICATION DES FICHIERS ===")
    for filename, description in data_files.items():
        filepath = Path(filename)
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {filename} ({description}) - {size_mb:.1f} MB")
        else:
            print(f"❌ {filename} - MANQUANT")
    print()

def quick_data_exploration():
    """Exploration rapide des données"""
    
    print("=== EXPLORATION RAPIDE ===")
    
    # Triplets
    try:
        triplets = pd.read_csv('train_triplet.txt', sep='\t', 
                              names=['user_id', 'song_id', 'play_count'], 
                              nrows=1000)  # Lire seulement 1000 lignes pour test
        print(f"Triplets - Exemple:")
        print(triplets.head(3))
        print(f"Shape: {triplets.shape}")
        print(f"Users uniques: {triplets['user_id'].nunique()}")
        print(f"Songs uniques: {triplets['song_id'].nunique()}")
        print()
        
    except Exception as e:
        print(f"❌ Erreur triplets: {e}")
    
    # Tracks
    try:
        tracks = pd.read_csv('p02_unique_tracks.txt', sep='<SEP>', 
                            names=['track_id', 'song_id', 'artist', 'title'],
                            engine='python', nrows=100)
        print(f"Tracks - Exemple:")
        print(tracks.head(3))
        print(f"Shape: {tracks.shape}")
        print()
        
    except Exception as e:
        print(f"❌ Erreur tracks: {e}")
    
    # Genres
    try:
        with open('p02_msd_tagtraum_cd2.cls.txt', 'r') as f:
            lines = [line.strip() for line in f.readlines()[:10] if line.strip() and not line.startswith('#')]
        
        print("Genres - Exemple:")
        for line in lines[:5]:
            print(line)
        print()
        
    except Exception as e:
        print(f"❌ Erreur genres: {e}")
    
    # Paroles
    try:
        with open('mxm_dataset_train.txt', 'r') as f:
            lines = [line.strip() for line in f.readlines()[:20]]
        
        print("Paroles - Exemple:")
        for line in lines[:5]:
            if not line.startswith('#'):
                print(line[:100] + "..." if len(line) > 100 else line)
        print()
        
    except Exception as e:
        print(f"❌ Erreur paroles: {e}")

def estimate_memory_usage():
    """Estime l'usage mémoire"""
    
    print("=== ESTIMATION MÉMOIRE ===")
    
    try:
        # Compter les lignes des triplets
        with open('train_triplet.txt', 'r') as f:
            line_count = sum(1 for line in f)
        
        # Estimation: ~50-100 bytes par ligne en mémoire
        estimated_mb = (line_count * 80) / (1024 * 1024)
        print(f"Triplets: ~{line_count:,} lignes")
        print(f"Mémoire estimée: ~{estimated_mb:.1f} MB")
        
        if estimated_mb > 4000:  # Plus de 4GB
            print("⚠️  ATTENTION: Dataset très volumineux!")
            print("   Considérez utiliser des chunks pour le chargement")
        elif estimated_mb > 1000:  # Plus de 1GB
            print("⚠️  Dataset moyen-volumineux, monitoring mémoire recommandé")
        else:
            print("✅ Taille de dataset gérable")
            
    except Exception as e:
        print(f"❌ Erreur estimation: {e}")

if __name__ == "__main__":
    check_data_files()
    quick_data_exploration()
    estimate_memory_usage()