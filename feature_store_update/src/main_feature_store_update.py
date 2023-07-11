from feature_store import FeatureStoreSupervisor

if __name__ == "__main__":
    fs = FeatureStoreSupervisor()
    fs.initialise_feature_store_update()
    print("Feature Store update complete")
