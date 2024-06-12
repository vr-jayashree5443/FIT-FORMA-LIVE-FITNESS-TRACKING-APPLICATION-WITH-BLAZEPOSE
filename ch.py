import warnings
import pickle
warnings.filterwarnings('ignore')
with open("input_scaler.pkl", "rb") as f:
    input_scaler = pickle.load(f)
    print("done")