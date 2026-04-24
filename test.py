import joblib

model = joblib.load('processed_data/final_xgboost_model.pkl')

print(type(model))

print(model.get_params())
