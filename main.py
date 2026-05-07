import joblib
import os
import pandas as pd
from app.utils import prepare_data
from app.engine import ModelTournament

def main():
    if not os.path.exists('models'): os.makedirs('models')
    
    # Ensure this path matches your file location
    df = pd.read_excel('data/excel.xlsx')
    states = df['State'].unique()
    registry = {}

    for state in states:
        print(f"Training Tournament for {state}...")
        data = prepare_data('data/excel.xlsx', state)
        
        if len(data) > 15:
            tournament = ModelTournament(data)
            registry[state] = tournament.run()
            print(f"Winner: {registry[state]['type']}")

    joblib.dump(registry, 'models/best_models.pkl')
    print("Tournament saved to models/best_models.pkl")

if __name__ == "__main__":
    main()