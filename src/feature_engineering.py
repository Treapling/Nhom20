from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    DROP_COLS,
    TARGET_COL,
    BasicHousingPreprocessor,
    ColumnDropper,
    build_preprocessing_pipeline,
    clip_series,
    fit_quantile_bounds,
    prepare_model_data,
)


def load_housing_data(path):
    return pd.read_csv(Path(path))


def make_clean_feature_table(df, target_col=TARGET_COL):
    preprocessor = BasicHousingPreprocessor(target_col=target_col)
    cleaned = preprocessor.fit_transform(df)
    preview_drop_cols = [col for col in DROP_COLS if col != "Location"] + ["price_per_m2"]
    cleaned = ColumnDropper(preview_drop_cols).fit_transform(cleaned)
    return cleaned, preprocessor


def split_raw_data(df, target_col=TARGET_COL, test_size=0.2, random_state=42):
    X, y = prepare_model_data(df, target_col=target_col)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def clip_train_test_target(y_train, y_test, lower_q=0.01, upper_q=0.99):
    target_bounds = fit_quantile_bounds(y_train, lower_q=lower_q, upper_q=upper_q)
    y_train_clipped = clip_series(y_train, target_bounds)
    y_test_clipped = clip_series(y_test, target_bounds)
    return y_train_clipped, y_test_clipped, target_bounds


def make_train_test_feature_tables(
    df,
    target_col=TARGET_COL,
    test_size=0.2,
    random_state=42,
):
    X_train, X_test, y_train, y_test = split_raw_data(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )
    y_train, y_test, target_bounds = clip_train_test_target(y_train, y_test)

    train_raw = X_train.copy()
    test_raw = X_test.copy()
    train_raw[target_col] = y_train
    test_raw[target_col] = y_test

    preprocessor = BasicHousingPreprocessor(target_col=target_col)
    train_features = preprocessor.fit_transform(train_raw)
    test_features = preprocessor.transform(test_raw)

    train_features = train_features.drop(columns=[target_col], errors="ignore")
    test_features = test_features.drop(columns=[target_col], errors="ignore")

    preview_drop_cols = [col for col in DROP_COLS if col != "Location"] + ["price_per_m2"]
    dropper = ColumnDropper(preview_drop_cols)
    train_features = dropper.transform(train_features)
    test_features = dropper.transform(test_features)

    preprocessor.external_target_bounds_ = target_bounds
    return train_features, test_features, y_train, y_test, preprocessor


def make_model_matrix(df, target_col=TARGET_COL):
    X, y = prepare_model_data(df, target_col=target_col)
    pipeline = build_preprocessing_pipeline(X)
    X_processed = pipeline.fit_transform(X, y)
    return X_processed, y, pipeline


def make_train_test_model_matrices(
    df,
    target_col=TARGET_COL,
    test_size=0.2,
    random_state=42,
):
    X_train, X_test, y_train, y_test = split_raw_data(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )
    y_train, y_test, _ = clip_train_test_target(y_train, y_test)
    pipeline = build_preprocessing_pipeline(X_train)
    X_train_processed = pipeline.fit_transform(X_train, y_train)
    X_test_processed = pipeline.transform(X_test)
    return X_train_processed, X_test_processed, y_train, y_test, pipeline


def export_all_processed_data(
    df,
    output_dir,
    target_col=TARGET_COL,
    test_size=0.2,
    random_state=42,
):
    """
    Xuất 4 file vào output_dir:
      - feature_engineered_train.csv   (readable, chưa encode/scale)
      - feature_engineered_test.csv
      - model_matrix_train.csv         (đã one-hot encode + StandardScaler)
      - model_matrix_test.csv
    Trả về dict chứa paths và các objects đã fit.
    """
    from pathlib import Path
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Feature tables (readable) ---
    train_df, test_df, y_train, y_test, preprocessor = make_train_test_feature_tables(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    train_export = train_df.copy()
    test_export = test_df.copy()
    train_export[target_col] = y_train.values
    test_export[target_col] = y_test.values

    feat_train_path = output_dir / "feature_engineered_train.csv"
    feat_test_path = output_dir / "feature_engineered_test.csv"
    train_export.to_csv(feat_train_path, index=False)
    test_export.to_csv(feat_test_path, index=False)

    # --- Model matrices (encoded + scaled) ---
    X_train_mat, X_test_mat, y_train_mat, y_test_mat, pipeline = make_train_test_model_matrices(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state,
    )

    # Lấy tên feature từ pipeline nếu có
    try:
        feature_names = pipeline.named_steps["encode_scale"].get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_mat.shape[1])]

    mat_train_df = pd.DataFrame(X_train_mat, columns=feature_names)
    mat_test_df = pd.DataFrame(X_test_mat, columns=feature_names)
    mat_train_df[target_col] = y_train_mat.values
    mat_test_df[target_col] = y_test_mat.values

    mat_train_path = output_dir / "model_matrix_train.csv"
    mat_test_path = output_dir / "model_matrix_test.csv"
    mat_train_df.to_csv(mat_train_path, index=False)
    mat_test_df.to_csv(mat_test_path, index=False)

    print(f"✅ Feature train : {feat_train_path}  {train_export.shape}")
    print(f"✅ Feature test  : {feat_test_path}  {test_export.shape}")
    print(f"✅ Matrix train  : {mat_train_path}  {mat_train_df.shape}")
    print(f"✅ Matrix test   : {mat_test_path}  {mat_test_df.shape}")

    return {
        "feature_train_path": feat_train_path,
        "feature_test_path": feat_test_path,
        "matrix_train_path": mat_train_path,
        "matrix_test_path": mat_test_path,
        "preprocessor": preprocessor,
        "pipeline": pipeline,
    }
