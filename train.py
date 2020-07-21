from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import VotingRegressor
from imbd.trainers import ModelTrainer
from imbd.data import DataLoader
from imbd.preprocessors import DataPreprocessor


def main():
    base_model = VotingRegressor([('xgb', XGBRegressor()),
                                  ('xgb_rf', XGBRFRegressor())])
    multi_output_model = MultiOutputRegressor(base_model)
    param_grid = {
        "prepro__variance_selector__threshold": [0.0, 0.01, 0.05],
        "model__estimator__n_estimators": [1000],
        "model__estimator__max_depth": [5, 10],
        # "model__estimator__alpha": [0, 0.1, 0.01],
        # "model__estimator__lambda": [1, 0.5, 0.1],
        "model__estimator__subsample": [1, 0.5],
        # "model__estimator__gamma": [0, 2, 10],
    }

    # initialization
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    df = loader.build()

    # get feature & label
    train_features = df.drop(loader.labels, axis=1)
    train_labels = df[loader.labels]

    # build pipeline
    steps = [('prepro', preprocessor), ('model', multi_output_model)]
    pipe = Pipeline(steps=steps)

    # training
    trainer = ModelTrainer(pipe=pipe, param_grid=param_grid, verbose=2)
    fitted = trainer.fit(train_features, train_labels)

    print(trainer.training_result)
    print(trainer.model.best_params_)
    print(trainer.model.best_score_)


if __name__ == '__main__':
    main()