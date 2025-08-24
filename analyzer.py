# simple_analyzer.py - Version simplifiée de l'analyseur MySpotify
import pandas as pd
import os
from pathlib import Path

def analyze_myspotify_results():
    """Analyse simple des résultats MySpotify"""
    
    results_path = Path("results")
    
    if not results_path.exists():
        print("❌ Dossier 'results' non trouvé.")
        print("   Exécutez d'abord: python complete_myspotify.py")
        return
    
    print("="*60)
    print("🎵 ANALYSE DES RÉSULTATS MYSPOTIFY 🎵")
    print("="*60)
    
    # Lister tous les fichiers CSV
    csv_files = list(results_path.glob("*.csv"))
    
    if not csv_files:
        print("❌ Aucun fichier CSV trouvé dans le dossier results/")
        return
    
    print(f"📁 Fichiers trouvés: {len(csv_files)}")
    print()
    
    total_recommendations = 0
    components_found = []
    
    # Analyser chaque fichier
    for file in sorted(csv_files):
        try:
            df = pd.read_csv(file)
            file_size_kb = file.stat().st_size / 1024
            
            print(f"📄 {file.name}")
            print(f"   • Taille: {file_size_kb:.1f} KB")
            print(f"   • Lignes: {len(df)}")
            print(f"   • Colonnes: {list(df.columns)}")
            
            # Échantillon des données
            if len(df) > 0:
                if 'artist' in df.columns and 'title' in df.columns:
                    print(f"   • Exemple: {df.iloc[0]['artist']} - {df.iloc[0]['title']}")
                total_recommendations += len(df)
            
            # Identifier le type de composant
            if file.name == "top_250_tracks.csv":
                components_found.append("✅ Top-250 tracks")
            elif file.name.startswith("top_100_"):
                genre = file.name.replace("top_100_", "").replace(".csv", "").title()
                components_found.append(f"✅ Top-100 {genre}")
            elif file.name.startswith("collection_"):
                theme = file.name.replace("collection_", "").replace(".csv", "").title()
                components_found.append(f"✅ Collection {theme}")
            elif file.name == "user_based_recommendations.csv":
                components_found.append("✅ User-based Collaborative Filtering")
            elif file.name == "item_based_recommendations.csv":
                components_found.append("✅ Item-based Collaborative Filtering")
            
            print()
            
        except Exception as e:
            print(f"❌ Erreur lecture {file.name}: {e}")
            print()
    
    # Résumé final
    print("="*60)
    print("📊 RÉSUMÉ FINAL")
    print("="*60)
    print(f"🎯 Total des recommandations: {total_recommendations:,}")
    print(f"📋 Composants implémentés: {len(components_found)}")
    print()
    
    print("🎵 Composants trouvés:")
    for component in components_found:
        print(f"   {component}")
    
    # Vérification des exigences projet
    print()
    print("📋 VÉRIFICATION EXIGENCES PROJET:")
    
    required_files = {
        "top_250_tracks.csv": "Top-250 tracks",
        "user_based_recommendations.csv": "User-based CF", 
        "item_based_recommendations.csv": "Item-based CF"
    }
    
    genre_files = [f for f in csv_files if f.name.startswith("top_100_")]
    collection_files = [f for f in csv_files if f.name.startswith("collection_")]
    
    # Vérifications
    all_good = True
    
    for req_file, description in required_files.items():
        if any(f.name == req_file for f in csv_files):
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description} - MANQUANT")
            all_good = False
    
    if len(genre_files) > 0:
        print(f"   ✅ Recommandations par genre ({len(genre_files)} genres)")
    else:
        print(f"   ❌ Recommandations par genre - MANQUANT")
        all_good = False
    
    if len(collection_files) > 0:
        print(f"   ✅ Collections thématiques ({len(collection_files)} thèmes)")
    else:
        print(f"   ❌ Collections thématiques - MANQUANT")
        all_good = False
    
    print()
    if all_good:
        print("🎉 FÉLICITATIONS! Tous les composants requis sont présents!")
        print("✅ Votre projet MySpotify est COMPLET!")
    else:
        print("⚠️  Certains composants manquent.")
        print("   Exécutez: python complete_myspotify.py")
    
    print("="*60)

def show_sample_recommendations():
    """Affiche des échantillons de recommandations"""
    
    print("\n🎵 ÉCHANTILLONS DE RECOMMANDATIONS")
    print("="*50)
    
    results_path = Path("results")
    
    # Top-250
    top_250_file = results_path / "top_250_tracks.csv"
    if top_250_file.exists():
        df = pd.read_csv(top_250_file)
        print("\n🏆 TOP 5 TRACKS LES PLUS POPULAIRES:")
        print("-" * 40)
        for i, row in df.head(5).iterrows():
            print(f"{row['rank']:2d}. {row['artist']} - {row['title']}")
            print(f"    Play count: {row['play_count']:,}")
    
    # Collection amour
    love_file = results_path / "collection_love.csv"
    if love_file.exists():
        df = pd.read_csv(love_file)
        print(f"\n💖 TOP 5 CHANSONS D'AMOUR:")
        print("-" * 40)
        for i, row in df.head(5).iterrows():
            print(f"{row['rank']:2d}. {row['artist']} - {row['title']}")
            if 'theme_score' in row:
                print(f"    Score thématique: {row['theme_score']}")
    
    # Rock
    rock_file = results_path / "top_100_rock.csv"
    if rock_file.exists():
        df = pd.read_csv(rock_file)
        print(f"\n🎸 TOP 5 ROCK:")
        print("-" * 40)
        for i, row in df.head(5).iterrows():
            print(f"{row['rank']:2d}. {row['artist']} - {row['title']}")
            print(f"    Play count: {row['play_count']:,}")

def main():
    """Fonction principale"""
    try:
        analyze_myspotify_results()
        show_sample_recommendations()
        
        print(f"\n📁 Pour voir tous les détails, consultez le dossier 'results/'")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()