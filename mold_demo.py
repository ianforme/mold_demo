import pandas as pd
import streamlit as st
import io
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# LOGIN
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

def generate_outliers(df, y_col_name, yhat_col_name, outlier_threshold):
    df = df.copy()
    df["error_abs"] = abs(df[yhat_col_name] - df[y_col_name])
    df["error_abs_pct"] = abs(df[yhat_col_name] - df[y_col_name]) / df[y_col_name]

    # Identify the outliers
    df["outlier"] = np.where(df["error_abs_pct"] <= outlier_threshold, 0, 1)
    return df


def model_eval(df_comparison, outlier_threshold, y_col_name, yhat_col_name):
    # Run `generate_outliers` before this function
    df = df_comparison[df_comparison["outlier"] == 0]
    outliers_cnt = df_comparison.shape[0] - df.shape[0]
    outliers_pct = outliers_cnt / df_comparison.shape[0]

    # Calculate the R2 score
    r2 = r2_score(df[y_col_name], df[yhat_col_name])

    # Calculate the mean absolute error
    mae = df["error_abs"].mean()

    # Calculate the mean absolute percentage error
    mape = df["error_abs_pct"].mean()

    global_r2 = r2_score(df_comparison[y_col_name], df_comparison[yhat_col_name])
    global_mape = df_comparison['error_abs_pct'].mean()

    print(
        f"Global R^2 Score: {global_r2}"
    )
    print(
        f"Global Mean Absolute Percentage Error (MAPE): {global_mape}%"
    )

    print("Selected absolute error % threshold:", outlier_threshold)
    print(f"R^2 Score: {r2}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape * 100}%")
    print(f"Number of records: {df_comparison.shape[0]}")
    print(f"Number of outliers: {outliers_cnt}")
    print(f"Percentage of outliers: {outliers_pct * 100}%")

    return global_r2, global_mape, r2, mae, mape


def process_data(df, model_cols, if_train=True):
    df = df[~df['品名'].isin(['Deco', 'MIDDLE_FRAME', 'Middle frame'])]

    series_dict = {
        'NAK80': ['進口NAK80', 'NAK80'],
        '718H': ['739H', '738H', '738', '718H'],
        'P20': ['P20HH', 'P20'],
        'LKM738': ['LKM838HS', 'LKM838H', 'LKM838', 'LKM738H', 'LKM738'],
        'S136': ['S136'],
        'KEYLOS2002'
        'SA710': ['SA715', 'SA710'],
        'NE-1030': ['NE-1030'],
        'STAVAX-S136': ['STAVAX-S136'],
        'K21353': ['K21353'],
        'LJ300ESR': ['LJ300ESR'],
        'ESKEYLO2002': ['ESKEYLO2002']
    }

    series_dict_proc = {}

    for k, v in series_dict.items():
        for i in v:
            series_dict_proc[i] = k

    common_material = ['NAK80', '718H', 'P20', 'LKM738']

    def proc_materials(s):

        all_s = []
        all_m = str(s).split('/')
        for m in all_m:
            if m in series_dict_proc.keys():
                series = series_dict_proc[m]
                if series not in common_material:
                    series = '非常见原料'
                all_s.append(series)

        return list(set(all_s))

    df['公仁材料系列'] = df['公仁材料'].apply(proc_materials)
    df['母仁材料系列'] = df['母仁材料'].apply(proc_materials)

    expanded_df = df['公仁材料系列'].apply(lambda lst: pd.Series(1, index=lst)).fillna(0).astype(int)
    expanded_df.columns = [f"公仁材料系列_{col}" for col in expanded_df.columns]

    expanded_df2 = df['母仁材料系列'].apply(lambda lst: pd.Series(1, index=lst)).fillna(0).astype(int)
    expanded_df2.columns = [f"母仁材料系列_{col}" for col in expanded_df2.columns]

    # Convert 咬花規格 text to common categories
    def proc_咬花規格(value):
        if pd.isna(value):
            return []
        all_matched_values = []
        value = value.replace("-", "").lower()
        for value_to_match in ['mt11015', 'mt11020', 'mt11010', 'mt11015', 'mt11000']:  # 特殊花紋
            if value_to_match in value:
                all_matched_values.append(value_to_match)
                value = value.replace(value_to_match, "")
        if len(value.replace("/", "").strip()) >= 1:
            all_matched_values.append('其他')
        return list(set(all_matched_values))

    df['咬花規格系列'] = df['咬花規格'].apply(proc_咬花規格)

    expanded_df3 = df['咬花規格系列'].apply(lambda lst: pd.Series(1, index=lst)).fillna(0).astype(int)
    expanded_df3.columns = [f"咬花規格系列_{col}" for col in expanded_df3.columns]

    df = df.join(expanded_df).join(expanded_df2).join(expanded_df3)
    df = df.drop(['公仁材料系列', '母仁材料系列', '公仁材料', '母仁材料', '咬花規格', '咬花規格系列', '品名'], axis=1)

    to_infer_cols = [i for i in model_cols if (i not in df.columns) and (i != target)]
    df[to_infer_cols] = 0

    if if_train:
        df = df[model_cols + [target]]
    else:
        df = df[model_cols]

    df = df.replace('-', np.nan)
    return df

def get_model_preds(model_type, train_X, train_y, test_X, model_params=dict()):
    if model_type == "linear_regression_default":
        model = LinearRegression(**model_params)
        model.fit(train_X.fillna(-1), train_y)
        y_pred = model.predict(test_X.fillna(-1))
    elif model_type == "random_forest":
        model = RandomForestRegressor(random_state=42, n_jobs=10, **model_params)
        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)
    elif model_type == "linear_regression_normalized":
        # pipeline that normalizes data, impute missing values with the mean, then trains a linear regression model
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("imputer", SimpleImputer(strategy="mean")),
                ("linear_regression", LinearRegression(**model_params)),
            ]
        )
        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)
    elif model_type == "elastic_net_normalized":
        # pipeline that normalizes data, impute missing values with the mean, then trains a linear regression model
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("imputer", SimpleImputer(strategy="mean")),
                ("elastic_net", ElasticNet(**model_params)),
            ]
        )
        model.fit(train_X, train_y)
        y_pred = model.predict(test_X)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    return model, y_pred


def get_predictions(df, cols_to_ignore, y_col_name, model_type, is_use_cv: bool, model_params=dict()):
    models_and_x_test = []
    df_with_preds = df.copy()
    split_indexes = KFold(n_splits=len(df)).split(df)
    progress_bar = st.progress(0)
    if is_use_cv:
        for idx, (train_index, test_index) in enumerate(tqdm(split_indexes, desc="Cross-validation")):
            progress_bar.progress(idx / len(df))
            # get actual index instead of positional index returned by KFols
            train_index = df.index.values[train_index]
            test_index = df.index.values[test_index]

            # Get train/test X/y
            train_df = df.loc[train_index]
            train_X = train_df.drop([*cols_to_ignore, y_col_name], axis=1)
            train_y = train_df[y_col_name]

            test_df = df.loc[test_index]
            test_X = test_df.drop([*cols_to_ignore, y_col_name], axis=1)
            test_y = test_df[y_col_name]

            # Train and predict
            model, y_pred = get_model_preds(model_type, train_X, train_y, test_X, model_params=model_params)

            # Append every single model and X test (for SHAP calculations)
            models_and_x_test.append({"model": model, "test_X": test_X, "train_X": train_X})
            # Append preds
            df_with_preds.loc[test_index, "preds"] = y_pred
    else:
        train_X, test_X, train_y, test_y = train_test_split(
            df.drop([*cols_to_ignore, y_col_name], axis=1),
            df[y_col_name],
            test_size=0.3,
            random_state=42,
        )
        model, y_pred = get_model_preds(model_type, train_X, train_y, test_X, model_params=model_params)
        df_with_preds = test_X.copy()
        df_with_preds[y_col_name] = test_y
        df_with_preds["preds"] = y_pred
        models_and_x_test.append({"model": model, "test_X": test_X, "train_X": train_X})
    progress_bar.progress(1.0)
    return df_with_preds, models_and_x_test


with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Create an authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)


name, authentication_status, username = authenticator.login('main')

if authentication_status:
    # App title
    st.title('模具建议价格智能生成')

    # File upload
    train_file = st.file_uploader("上传模型训练数据", type=["xlsx"])
    if train_file is not None:
        train_data = pd.read_excel(train_file, sheet_name='Sheet1')
        st.dataframe(train_data)

    test_file = st.file_uploader("上传需建议模具数据", type=["xlsx"])
    if test_file is not None:
        test_data = pd.read_excel(test_file, sheet_name='Sheet1')
        st.dataframe(test_data)

    # Run button
    if st.button("训练模型，建议价格"):
        if train_file is not None and test_file is not None:
            useful_cols = [
                'material_ID', '品名', '寸别', '取模時間', '成形時間', '咬花規格',
                '成品体積', '成品重量', '建議熱膠道數量', '成品尺寸-L', '成品尺寸-W',
                '成品尺寸-H',
                # '成品尺寸-T',
                '成品外觀_咬花', '模具形式_倒灌模',
                # '摊提数量',
                '母仁材料', '公仁材料',
                '客户类型_OBM',
                '客户类型_客指客供',
                '客户类型_普通客户'

            ]

            model_cols = [
                'material_ID', '寸别', '取模時間', '成形時間', '成品体積', '成品重量',
                '建議熱膠道數量', '成品尺寸-L', '成品尺寸-W', '成品尺寸-H',
                # '成品尺寸-T',
                '成品外觀_咬花', '模具形式_倒灌模',
                # '摊提数量',
                '公仁材料系列_718H', '公仁材料系列_LKM738', '公仁材料系列_P20',
                '公仁材料系列_NAK80', '公仁材料系列_非常见原料', '母仁材料系列_LKM738',
                '母仁材料系列_718H', '母仁材料系列_NAK80', '母仁材料系列_非常见原料',
                '母仁材料系列_P20', '咬花規格系列_mt11015', '咬花規格系列_其他',
                '咬花規格系列_mt11020', '咬花規格系列_mt11000', '咬花規格系列_mt11010',
                '客户类型_OBM',
                '客户类型_客指客供',
                '客户类型_普通客户'
            ]

            target = 'mold_cost_usd'
            id_col = 'material_ID'
            model_params = {}
            outlier_threshold = 0.3

            train_data_proc = process_data(train_data[useful_cols + [target]], model_cols)
            test_data_proc = process_data(test_data[useful_cols], model_cols, False)

            # Progress bar
            with st.expander("训练模型...", expanded=True):
                progress_bar = st.progress(0)
                model = RandomForestRegressor(random_state=42, n_jobs=10, **model_params)
                n_estimators = model.get_params().get('n_estimators', 100)

                for i in tqdm(range(1, n_estimators + 1), desc="Training model"):
                    progress_bar.progress(i / n_estimators)
                    model.set_params(n_estimators=i)
                    model.fit(train_data_proc[model_cols].drop(id_col, axis=1), train_data_proc[target])

            with st.expander("验证模型...", expanded=True):
                df_with_preds, models_and_x_test = get_predictions(train_data_proc, [id_col], target, 'random_forest',
                                                                   is_use_cv=True, model_params=model_params)
                df_with_preds = generate_outliers(df_with_preds, "mold_cost_usd", "preds", outlier_threshold)
                global_r2, global_mape, r2, mae, mape = model_eval(df_with_preds, outlier_threshold, "mold_cost_usd", "preds")

                st.write(f"全局R2: {global_r2:.2f}")
                st.write(f"全局MAPE: {global_mape:.1%}")
                st.write(f"常规值R2: {r2:.2f}")
                st.write(f"常规值MAPE: {mape:.1%}")

            with st.expander("生成建议价格...", expanded=True):
                model_preds = model.predict(test_data_proc[model_cols].drop(id_col, axis=1))

                output = test_data[useful_cols].copy()
                output['建议价格'] = model_preds
                output['建议价格'] = output['建议价格'].round(2)
                st.dataframe(output)

                # Convert DataFrame to Excel
                output_io = io.BytesIO()
                with pd.ExcelWriter(output_io, engine='xlsxwriter') as writer:
                    output.to_excel(writer, index=False, sheet_name='Sheet1')
                output_io.seek(0)

                st.download_button(
                    label="下载建议数据",
                    data=output_io,
                    file_name="模具建议价格.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.write("请上传模型训练数据与模具建议数据.")

elif authentication_status == False:
    st.error('Username/password is incorrect')

elif authentication_status is None:
    st.warning('Please enter your username and password')
