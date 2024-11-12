import pickle

def dump(file_name, data, protocol=3):
    with open(file_name, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
def load_pkl(file_name):
    with open(file_name, "rb") as f:
        try:
            return pickle.load(f)
        except Exception as exc:
            import pickle5 as pickle
            return pickle.load(f)

