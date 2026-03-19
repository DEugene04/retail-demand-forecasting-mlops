from src.data.load_data import load_dataset
from pipelines.build_features import build_train_test_features
from src.models.train import run_baseline_model

holiday, oil, train, test, stores, transactions = load_dataset()

train_encoded, test_encoded = build_train_test_features(
    train=train,
    test=test,
    oil=oil,
    stores=stores,
    holiday=holiday
)

model, rmsle = run_baseline_model(train_encoded)