from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from imbd.trainers import ModelTrainer
from imbd.data import DataLoader
from imbd.preprocessors import DataPreprocessor


def main():
    base_model = MultiOutputRegressor(XGBRegressor())
    param_grid = {
        "prepro__variance_selector__threshold": [0.0, 0.01, 0.1],
        "model__estimator__n_estimators": [100, 1000],
        "model__estimator__max_depth": [2, 5, 10],
        "model__estimator__alpha": [0, 0.1, 0.01],
        "model__estimator__lambda": [1, 0.5, 0.1],
        "model__estimator__subsample": [1, 0.5, 0.1],
        "model__estimator__gamma": [0, 2, 10],
    }

    # initialization
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    df = loader.build()

    # get feature & label
    train_features = df.drop(loader.labels, axis=1)
    train_labels = df[loader.labels]

    # build pipeline
    steps = [('prepro', preprocessor), ('model', base_model)]
    pipe = Pipeline(steps=steps)

    # training
    trainer = ModelTrainer(pipe=pipe, param_grid=param_grid, verbose=True)
    fitted = trainer.train(train_features, train_labels)

    print(trainer.training_result)


if __name__ == '__main__':
    main()