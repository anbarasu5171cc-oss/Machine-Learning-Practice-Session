import hashlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ML Chat Workspace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
<style>
:root{
    --bg:#07111c;
    --panel:#0e1b2d;
    --panel2:#13243c;
    --line:#34577f;
    --text:#edf5ff;
    --muted:#a4bdd8;
    --accent:#4fb7ff;
    --accent2:#8b7dff;
    --accent3:#31d59a;
    --accent4:#ffbf47;
    --danger:#ff7082;
}
html, body, [class*="css"]{
    background:var(--bg);
    color:var(--text);
    font-family:"Inter", sans-serif;
    font-size:13px !important;
}
#MainMenu, footer, header{visibility:hidden;}
[data-testid="stAppViewContainer"]{
    background:
        radial-gradient(circle at top left, rgba(79,183,255,0.10), transparent 24%),
        radial-gradient(circle at top right, rgba(139,125,255,0.10), transparent 24%),
        linear-gradient(180deg, #08131f 0%, #07111c 100%);
}
.block-container{
    max-width:100% !important;
    padding-top:0.55rem !important;
    padding-bottom:0.55rem !important;
}
.main-title{
    background:linear-gradient(135deg, rgba(79,183,255,0.16), rgba(139,125,255,0.15));
    border:1px solid rgba(79,183,255,0.25);
    border-radius:20px;
    padding:1px 16px;
    margin-bottom:12px;
}
.main-title h1{
    margin:0;
    color:#ffffff;
    font-size:22px;
}
.main-title p{
    margin:6px 0 0 0;
    color:var(--muted);
    font-size:11px;
}
.panel-title{
    font-size:18px;
    font-weight:700;
    color:#ffffff;
    margin-bottom:2px;
}
.panel-subtitle{
    color:var(--muted);
    font-size:11px;
    margin-bottom:10px;
}
.command-card{
    background:#112339;
    border:1px solid #3a618e;
    border-radius:12px;
    padding:8px 10px;
    margin-bottom:6px;
}
.command-name{
    color:#ffffff;
    font-size:12px;
    font-weight:700;
}
.command-desc{
    color:#d4e6f8;
    font-size:11px;
    margin-top:1px;
}
.info-box{
    background:#122744;
    border:1px solid #2c648f;
    color:#e5f2ff;
    border-radius:12px;
    padding:8px 10px;
    margin-bottom:8px;
    font-size:12px;
}
.success-box{
    background:#123427;
    border:1px solid #2d8a67;
    color:#d9ffeb;
    border-radius:12px;
    padding:8px 10px;
    margin-bottom:8px;
    font-size:12px;
}
.warn-box{
    background:#3b2b13;
    border:1px solid #9f7130;
    color:#ffe9be;
    border-radius:12px;
    padding:8px 10px;
    margin-bottom:8px;
    font-size:12px;
}
[data-testid="stButton"] > button{
    width:100%;
    min-height:36px;
    border-radius:12px !important;
    border:1px solid #3b628f !important;
    background:#132540 !important;
    color:#ffffff !important;
    font-weight:600 !important;
    font-size:12px !important;
}
[data-testid="stButton"] > button:hover{
    background:#1a3456 !important;
    border-color:#4fb7ff !important;
}
[data-testid="stTextInput"] input,
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stNumberInput input{
    background:#112238 !important;
    color:#ffffff !important;
    border:1px solid #3a608d !important;
    border-radius:10px !important;
    font-size:12px !important;
}
[data-testid="stFileUploader"]{
    background:#112238;
    border:1px dashed #456c9c;
    border-radius:14px;
    padding:6px;
}
[data-testid="stExpander"]{
    border:1px solid #33567e;
    border-radius:14px;
}
[data-testid="stMetricValue"]{
    color:#ffffff;
    font-size:20px !important;
}
[data-testid="stMetricLabel"]{
    font-size:11px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# MATPLOTLIB STYLE
# =========================================================
plt.rcParams.update(
    {
        "figure.facecolor": "#152842",
        "axes.facecolor": "#152842",
        "axes.edgecolor": "#678cba",
        "axes.labelcolor": "#edf5ff",
        "xtick.color": "#d8e8ff",
        "ytick.color": "#d8e8ff",
        "text.color": "#f7fbff",
        "grid.color": "#6989ae",
        "legend.facecolor": "#152842",
        "legend.edgecolor": "#678cba",
    }
)

# =========================================================
# CONSTANTS
# =========================================================
PAGE_ORDER = [
    "Upload",
    "Preview",
    "Encoding",
    "Problem Type",
    "Target & Features",
    "Preprocess",
    "Train",
    "Results",
    "Visualization",
]

COMMAND_DEFINITIONS = [
    ("upload", "Open the dataset upload page."),
    ("preview", "Show original dataset preview and summary."),
    ("encoding", "Show original dataset, fully encoded dataset, and encoding mappings."),
    ("problem type", "Choose classification, regression, or clustering."),
    ("target", "Select target column and input features."),
    ("preprocess", "Configure scaling, encoding usage, and PCA."),
    ("train", "Choose algorithm and train the model."),
    ("results", "Show metrics, evaluation plots, and prediction output."),
    ("visualization", "Open multiple custom visualization plots."),
]

CLASSIFICATION_ALGOS = [
    "Logistic Regression",
    "Decision Tree Classifier",
    "Random Forest Classifier",
    "Bagging Classifier",
    "AdaBoost Classifier",
    "Gradient Boosting Classifier",
    "SVM Classifier",
    "KNN Classifier",
]

REGRESSION_ALGOS = [
    "Linear Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Bagging Regressor",
    "AdaBoost Regressor",
    "Gradient Boosting Regressor",
    "SVM Regressor",
    "KNN Regressor",
]

CLUSTERING_ALGOS = [
    "K-Means",
    "DBSCAN",
    "Agglomerative Clustering",
]

ALGO_INFO = {
    "Logistic Regression": "Linear classification model.",
    "Decision Tree Classifier": "Rule-based classifier.",
    "Random Forest Classifier": "Tree ensemble classifier.",
    "Bagging Classifier": "Bootstrap ensemble classifier.",
    "AdaBoost Classifier": "Boosting classifier.",
    "Gradient Boosting Classifier": "Sequential boosting classifier.",
    "SVM Classifier": "Margin-based classifier.",
    "KNN Classifier": "Nearest-neighbor classifier.",
    "Linear Regression": "Linear model for continuous prediction.",
    "Decision Tree Regressor": "Rule-based regressor.",
    "Random Forest Regressor": "Tree ensemble regressor.",
    "Bagging Regressor": "Bootstrap ensemble regressor.",
    "AdaBoost Regressor": "Boosting regressor.",
    "Gradient Boosting Regressor": "Sequential boosting regressor.",
    "SVM Regressor": "Support vector regressor.",
    "KNN Regressor": "Nearest-neighbor regressor.",
    "K-Means": "Distance-based clustering.",
    "DBSCAN": "Density-based clustering.",
    "Agglomerative Clustering": "Hierarchical clustering.",
}

# =========================================================
# SESSION STATE
# =========================================================
def init_state() -> None:
    defaults = {
        "active_page": "Upload",
        "unlocked_pages": ["Upload"],
        "uploaded_file_hash": None,
        "trained": False,
        "results": {},
        "problem_choice": "Classification",
        "target_col": None,
        "selected_features": [],
        "ignored_non_numeric": [],
        "scaler_choice": "StandardScaler",
        "use_pca": False,
        "pca_components": 2,
        "test_size": 0.2,
        "algo_cls": "Logistic Regression",
        "algo_reg": "Linear Regression",
        "algo_clu": "K-Means",
        "k_clusters": 3,
        "dbscan_eps": 0.5,
        "dbscan_min_samples": 5,
        "rare_class_warning": "",
        "model_bundle": None,
        "encoding_mappings": {},
        "encoding_enabled": True,
        "pending_command": "",
        "viz_choices": {
            "line_x": None,
            "line_y": None,
            "bar_x": None,
            "bar_y": None,
            "scatter_x": None,
            "scatter_y": None,
            "box_x": None,
            "box_y": None,
            "violin_x": None,
            "violin_y": None,
            "area_x": None,
            "area_y": None,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# =========================================================
# STATE HELPERS
# =========================================================
def reset_for_new_dataset() -> None:
    st.session_state.active_page = "Upload"
    st.session_state.unlocked_pages = ["Upload"]
    st.session_state.trained = False
    st.session_state.results = {}
    st.session_state.problem_choice = "Classification"
    st.session_state.target_col = None
    st.session_state.selected_features = []
    st.session_state.ignored_non_numeric = []
    st.session_state.scaler_choice = "StandardScaler"
    st.session_state.use_pca = False
    st.session_state.pca_components = 2
    st.session_state.test_size = 0.2
    st.session_state.algo_cls = "Logistic Regression"
    st.session_state.algo_reg = "Linear Regression"
    st.session_state.algo_clu = "K-Means"
    st.session_state.k_clusters = 3
    st.session_state.dbscan_eps = 0.5
    st.session_state.dbscan_min_samples = 5
    st.session_state.rare_class_warning = ""
    st.session_state.model_bundle = None
    st.session_state.encoding_mappings = {}
    st.session_state.encoding_enabled = True
    st.session_state.pending_command = ""
    st.session_state.viz_choices = {
        "line_x": None,
        "line_y": None,
        "bar_x": None,
        "bar_y": None,
        "scatter_x": None,
        "scatter_y": None,
        "box_x": None,
        "box_y": None,
        "violin_x": None,
        "violin_y": None,
        "area_x": None,
        "area_y": None,
    }


def get_file_hash(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def unlock_page(page_name: str) -> None:
    if page_name not in PAGE_ORDER:
        return
    idx = PAGE_ORDER.index(page_name)
    st.session_state.unlocked_pages = PAGE_ORDER[: idx + 1]
    st.session_state.active_page = page_name


def set_active_page(page_name: str) -> None:
    if page_name in st.session_state.unlocked_pages:
        st.session_state.active_page = page_name


def parse_command(command: str) -> Optional[str]:
    raw = command.strip().lower()
    mapping = {
        "upload": "Upload",
        "preview": "Preview",
        "encoding": "Encoding",
        "problem type": "Problem Type",
        "problem": "Problem Type",
        "target": "Target & Features",
        "features": "Target & Features",
        "preprocess": "Preprocess",
        "train": "Train",
        "results": "Results",
        "result": "Results",
        "visualization": "Visualization",
        "visualisation": "Visualization",
    }
    for key, value in mapping.items():
        if key in raw:
            return value
    return None


def handle_chat_command(command: str) -> None:
    if not command.strip():
        return
    page = parse_command(command)
    if page is not None:
        unlock_page(page)


# =========================================================
# DATA HELPERS
# =========================================================
def make_unique_columns(columns) -> list[str]:
    counts = {}
    new_cols = []
    for col in columns:
        col_str = str(col).strip()
        if col_str == "":
            col_str = "column"
        if col_str not in counts:
            counts[col_str] = 0
            new_cols.append(col_str)
        else:
            counts[col_str] += 1
            new_cols.append(f"{col_str}_{counts[col_str]}")
    return new_cols


def get_df(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        df.columns = make_unique_columns(df.columns)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]
        return df
    except Exception as exc:
        st.error(f"CSV read error: {exc}")
        return None


def encode_full_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    encoded_df = df.copy()
    mappings = {}

    for col in encoded_df.columns:
        if encoded_df[col].dtype == "object" or str(encoded_df[col].dtype) == "category" or pd.api.types.is_bool_dtype(encoded_df[col]):
            series = encoded_df[col].astype(str).fillna("missing")
            uniques = pd.Series(series).unique().tolist()
            mapping = {value: idx for idx, value in enumerate(uniques)}
            mappings[col] = mapping
            encoded_df[col] = series.map(mapping).astype(int)

    return encoded_df, mappings


def infer_problem_type(y: pd.Series) -> str:
    if y.dtype == "object" or str(y.dtype) == "category" or pd.api.types.is_bool_dtype(y):
        return "Classification"
    unique_count = y.nunique(dropna=True)
    total_count = len(y)
    if unique_count <= min(20, max(2, int(total_count * 0.05))):
        return "Classification"
    return "Regression"


def get_numeric_feature_candidates(df: pd.DataFrame, target_col: Optional[str] = None) -> tuple[list[str], list[str]]:
    cols = [c for c in df.columns if c != target_col]
    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    ignored_cols = [c for c in cols if c not in numeric_cols]
    return numeric_cols, ignored_cols


def build_classification_model(name: str):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        "Random Forest Classifier": RandomForestClassifier(random_state=42),
        "Bagging Classifier": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=25,
            random_state=42,
        ),
        "AdaBoost Classifier": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
        "SVM Classifier": SVC(probability=True),
        "KNN Classifier": KNeighborsClassifier(),
    }
    return models[name]


def build_regression_model(name: str):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Bagging Regressor": BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=42),
            n_estimators=25,
            random_state=42,
        ),
        "AdaBoost Regressor": AdaBoostRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        "SVM Regressor": SVR(),
        "KNN Regressor": KNeighborsRegressor(),
    }
    return models[name]


def preprocess_fit(
    X_df: pd.DataFrame,
    scaler_choice: str = "StandardScaler",
    use_pca: bool = False,
    pca_components: int = 2,
):
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_df), columns=X_df.columns)

    scaler = None
    X_scaled = X_imputed.values

    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    elif scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_imputed)

    pca_model = None
    X_final = X_scaled
    if use_pca:
        max_components = min(max(1, pca_components), X_scaled.shape[1], X_scaled.shape[0])
        pca_model = PCA(n_components=max_components)
        X_final = pca_model.fit_transform(X_scaled)

    return X_final, imputer, scaler, pca_model


def transform_manual_input(
    input_df: pd.DataFrame,
    imputer,
    scaler=None,
    pca_model=None,
) -> np.ndarray:
    X_imp = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    X_val = X_imp.values
    if scaler is not None:
        X_val = scaler.transform(X_imp)
    if pca_model is not None:
        X_val = pca_model.transform(X_val)
    return X_val


def encode_class_target(y_series: pd.Series):
    unique_classes = list(pd.Series(y_series.astype(str)).dropna().unique())
    mapping = {label: idx for idx, label in enumerate(unique_classes)}
    reverse_mapping = {v: k for k, v in mapping.items()}
    y_numeric = y_series.astype(str).map(mapping).values
    return y_numeric, reverse_mapping


def safe_stratify_target(y: np.ndarray):
    counts = pd.Series(y).value_counts()
    rare = counts[counts < 2]
    if len(rare) > 0:
        return None, "Rare classes detected. Stratified split was disabled because some target values appear only once."
    return y, ""


# =========================================================
# PLOT HELPERS
# =========================================================
def show_plot(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_actual_vs_pred(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    n = min(50, len(y_true))
    ax.plot(range(n), np.array(y_true)[:n], color="#52b4ff", linewidth=2.2, label="Actual")
    ax.plot(range(n), np.array(y_pred)[:n], color="#ffbf47", linewidth=2.2, linestyle="--", label="Predicted")
    ax.set_title("Actual vs Predicted", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_class_distribution(preds, reverse_mapping=None):
    values, counts = np.unique(preds, return_counts=True)
    labels = [reverse_mapping.get(int(v), str(v)) if reverse_mapping else str(v) for v in values]
    colors = ["#52b4ff", "#8a77ff", "#33d094", "#ffbf47", "#ff6e7f", "#7fe7d7"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(labels, counts, color=colors[: len(labels)])
    ax.set_title("Prediction Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


def plot_regression_distribution(preds):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    sample = np.array(preds[:50])
    ax.plot(range(len(sample)), sample, marker="o", linewidth=2.2, color="#52b4ff")
    ax.set_title("Predicted Values", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Predicted Value")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_clusters(X_final, labels, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = np.array(labels)
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        name = "Noise" if lab == -1 else f"Cluster {lab}"
        ax.scatter(
            X_final[mask, 0],
            X_final[mask, 1],
            s=45,
            alpha=0.85,
            color=cmap(i),
            label=name,
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def safe_xy_df(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    temp = pd.DataFrame({
        "_x": df[x_col],
        "_y": df[y_col],
    }).dropna()
    return temp


def plot_line_chart(df: pd.DataFrame, x_col: str, y_col: str):
    data = safe_xy_df(df, x_col, y_col).sort_values(by="_x")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(data["_x"], data["_y"], color="#52b4ff", linewidth=2.2, marker="o")
    ax.set_title(f"Line Chart: {x_col} vs {y_col}", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_bar_chart(df: pd.DataFrame, x_col: str, y_col: str):
    data = safe_xy_df(df, x_col, y_col).head(25)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(data["_x"].astype(str), data["_y"], color="#8a77ff")
    ax.set_title(f"Bar Chart: {x_col} vs {y_col}", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig


def plot_scatter_chart(df: pd.DataFrame, x_col: str, y_col: str):
    data = safe_xy_df(df, x_col, y_col)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.scatter(data["_x"], data["_y"], color="#33d094", alpha=0.8, s=35)
    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_box_chart(df: pd.DataFrame, x_col: str, y_col: str):
    data = safe_xy_df(df, x_col, y_col)
    data["_x"] = data["_x"].astype(str)
    top_groups = data["_x"].value_counts().head(8).index
    data = data[data["_x"].isin(top_groups)]
    grouped = [data[data["_x"] == grp]["_y"].values for grp in top_groups]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bp = ax.boxplot(grouped, labels=list(top_groups), patch_artist=True)
    colors = ["#52b4ff", "#8a77ff", "#33d094", "#ffbf47", "#ff6e7f", "#7fe7d7", "#ffc7de", "#98b6ff"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title(f"Box Plot: {x_col} vs {y_col}", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    return fig


def plot_violin_chart(df: pd.DataFrame, x_col: str, y_col: str):
    data = safe_xy_df(df, x_col, y_col)
    data["_x"] = data["_x"].astype(str)
    top_groups = data["_x"].value_counts().head(8).index
    data = data[data["_x"].isin(top_groups)]
    grouped = [data[data["_x"] == grp]["_y"].values for grp in top_groups]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    parts = ax.violinplot(grouped, showmeans=True, showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor("#52b4ff")
        body.set_edgecolor("#dff4ff")
        body.set_alpha(0.75)
    ax.set_xticks(range(1, len(top_groups) + 1))
    ax.set_xticklabels(list(top_groups), rotation=20, ha="right")
    ax.set_title(f"Violin Plot: {x_col} vs {y_col}", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def plot_area_chart(df: pd.DataFrame, x_col: str, y_col: str):
    data = safe_xy_df(df, x_col, y_col).sort_values(by="_x")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.fill_between(data["_x"], data["_y"], color="#8a77ff", alpha=0.5)
    ax.plot(data["_x"], data["_y"], color="#8a77ff", linewidth=2)
    ax.set_title(f"Area Chart: {x_col} vs {y_col}", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# =========================================================
# TITLE
# =========================================================
st.markdown(
    """
<div class="main-title">
    <h1>🤖 ML Chat Workspace</h1>
    <p>Commands unlock steps one by one. Encoding concept is included as a separate step.</p>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# FILE INPUT
# =========================================================
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"], key="file_uploader_main")
current_hash = get_file_hash(uploaded_file)

if current_hash != st.session_state.uploaded_file_hash:
    st.session_state.uploaded_file_hash = current_hash
    if current_hash is not None:
        reset_for_new_dataset()

original_df = get_df(uploaded_file)
encoded_df = None

if original_df is not None:
    encoded_df, mappings = encode_full_dataset(original_df)
    st.session_state.encoding_mappings = mappings

    if st.session_state.target_col is None or st.session_state.target_col not in encoded_df.columns:
        st.session_state.target_col = encoded_df.columns[-1]

    numeric_candidates, ignored_cols = get_numeric_feature_candidates(encoded_df, st.session_state.target_col)
    st.session_state.ignored_non_numeric = ignored_cols

    if not st.session_state.selected_features:
        st.session_state.selected_features = numeric_candidates[:]

# =========================================================
# LAYOUT
# =========================================================
chat_col, output_col = st.columns([1, 3], gap="large")

# =========================================================
# LEFT PANEL
# =========================================================
with chat_col:
    st.markdown('<div class="panel-title">Command Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Enter one command only. Previous commands are hidden.</div>', unsafe_allow_html=True)

    for name, desc in COMMAND_DEFINITIONS:
        st.markdown(
            f"""
            <div class="command-card">
                <div class="command-name">{name}</div>
                <div class="command-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.text_input(
        "Type command",
        placeholder="upload / preview / encoding / problem type / train / results / visualization",
        key="chat_command_input",
    )

    if st.button("Send Command", key="send_command_button"):
        typed_command = st.session_state.chat_command_input
        handle_chat_command(typed_command)
        st.rerun()

# =========================================================
# RIGHT PANEL
# =========================================================
with output_col:
    st.markdown('<div class="panel-title">Output Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-subtitle">Steps unlock one by one</div>', unsafe_allow_html=True)

    tab_cols = st.columns(len(PAGE_ORDER))
    for i, page in enumerate(PAGE_ORDER):
        with tab_cols[i]:
            disabled_state = page not in st.session_state.unlocked_pages
            if st.button(page, key=f"tab_{page}", disabled=disabled_state):
                set_active_page(page)
                st.rerun()

    st.divider()
    active_page = st.session_state.active_page

    if active_page == "Upload":
        st.markdown("## Step 1 — Upload Dataset Process")

        if original_df is None:
            st.markdown(
                '<div class="info-box">Upload a CSV dataset first. Then enter commands to unlock the next steps.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="success-box">Dataset loaded successfully.</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", original_df.shape[0])
            c2.metric("Columns", original_df.shape[1])
            c3.metric("Encoded Columns", encoded_df.shape[1] if encoded_df is not None else 0)
            c4.metric("Target", str(st.session_state.target_col))

            st.markdown("### Original Dataset Preview")
            st.dataframe(original_df.head(8), use_container_width=True)

    elif active_page == "Preview":
        st.markdown("## Step 2 — Dataset Preview Process")

        if original_df is None:
            st.markdown('<div class="warn-box">Upload a dataset first.</div>', unsafe_allow_html=True)
        else:
            left, right = st.columns(2)

            with left:
                st.markdown("### Dataset Preview")
                st.dataframe(original_df.head(10), use_container_width=True)

            with right:
                st.markdown("### Summary")
                st.metric("Rows", original_df.shape[0])
                st.metric("Columns", original_df.shape[1])
                st.metric("Duplicate Rows", int(original_df.duplicated().sum()))
                st.metric("Missing Cells", int(original_df.isnull().sum().sum()))

            with st.expander("Missing Values", expanded=True):
                nulls = original_df.isnull().sum()
                nulls = nulls[nulls > 0]
                if len(nulls) == 0:
                    st.success("No missing values.")
                else:
                    st.dataframe(
                        nulls.reset_index().rename(columns={"index": "Column", 0: "Null Count"}),
                        use_container_width=True,
                    )

    elif active_page == "Encoding":
        st.markdown("## Step 3 — Encoding Concept Process")

        if original_df is None or encoded_df is None:
            st.markdown('<div class="warn-box">Upload a dataset first.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="info-box">Encoding converts categorical text values into numeric values so the machine learning model can process the full dataset.</div>',
                unsafe_allow_html=True,
            )

            left, right = st.columns(2)
            with left:
                st.markdown("### Original Dataset")
                st.dataframe(original_df.head(10), use_container_width=True)

            with right:
                st.markdown("### Fully Encoded Dataset")
                st.dataframe(encoded_df.head(10), use_container_width=True)

            with st.expander("Encoding Mappings", expanded=False):
                if not st.session_state.encoding_mappings:
                    st.write("No categorical columns required encoding.")
                else:
                    rows = []
                    for col, mapping in st.session_state.encoding_mappings.items():
                        for original_value, encoded_value in mapping.items():
                            rows.append(
                                {
                                    "Column": col,
                                    "Original Value": original_value,
                                    "Encoded Value": encoded_value,
                                }
                            )
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    elif active_page == "Problem Type":
        st.markdown("## Step 4 — Problem Type Detection Process")

        if encoded_df is None:
            st.markdown('<div class="warn-box">Upload a dataset first.</div>', unsafe_allow_html=True)
        else:
            target_col = st.session_state.target_col if st.session_state.target_col in encoded_df.columns else encoded_df.columns[-1]
            inferred = infer_problem_type(encoded_df[target_col])

            st.markdown(
                '<div class="info-box">The system suggests a learning problem from the current target column. You can change it manually.</div>',
                unsafe_allow_html=True,
            )

            options = ["Classification", "Regression", "Clustering"]
            current_problem = st.session_state.problem_choice if st.session_state.problem_choice in options else inferred

            st.session_state.problem_choice = st.radio(
                "Choose final problem type",
                options,
                index=options.index(current_problem),
                horizontal=True,
                key="problem_type_radio",
            )

            a, b, c = st.columns(3)
            a.metric("Target Column", target_col)
            b.metric("Auto Suggestion", inferred)
            c.metric("Unique Target Values", int(encoded_df[target_col].nunique(dropna=True)))

    elif active_page == "Target & Features":
        st.markdown("## Step 5 — Target and Feature Selection Process")

        if encoded_df is None:
            st.markdown('<div class="warn-box">Upload a dataset first.</div>', unsafe_allow_html=True)
        else:
            current_target = st.session_state.target_col if st.session_state.target_col in encoded_df.columns else encoded_df.columns[-1]

            new_target = st.selectbox(
                "Select target column",
                encoded_df.columns,
                index=list(encoded_df.columns).index(current_target),
                key="target_selectbox",
            )

            if new_target != st.session_state.target_col:
                st.session_state.target_col = new_target
                numeric_candidates, ignored_cols = get_numeric_feature_candidates(encoded_df, st.session_state.target_col)
                st.session_state.ignored_non_numeric = ignored_cols
                st.session_state.selected_features = numeric_candidates[:]
                st.session_state.trained = False
                st.session_state.results = {}
                st.session_state.model_bundle = None

            numeric_candidates, ignored_cols = get_numeric_feature_candidates(encoded_df, st.session_state.target_col)
            default_features = [f for f in st.session_state.selected_features if f in numeric_candidates] or numeric_candidates

            st.session_state.selected_features = st.multiselect(
                "Select input features from encoded dataset",
                numeric_candidates,
                default=default_features,
                key="feature_multiselect",
            )

            if ignored_cols:
                st.markdown(
                    f'<div class="warn-box">Ignored non-numeric feature columns: {", ".join(ignored_cols)}</div>',
                    unsafe_allow_html=True,
                )

    elif active_page == "Preprocess":
        st.markdown("## Step 6 — Preprocessing Process")

        if encoded_df is None:
            st.markdown('<div class="warn-box">Upload a dataset first.</div>', unsafe_allow_html=True)
        else:
            left, right = st.columns(2)

            with left:
                st.session_state.encoding_enabled = st.checkbox(
                    "Use encoded dataset for training",
                    value=st.session_state.encoding_enabled,
                    key="encoding_enabled_check",
                )

                scaler_options = ["StandardScaler", "MinMaxScaler", "None"]
                current_scaler = (
                    st.session_state.scaler_choice
                    if st.session_state.scaler_choice in scaler_options
                    else "StandardScaler"
                )

                st.session_state.scaler_choice = st.selectbox(
                    "Scaler",
                    scaler_options,
                    index=scaler_options.index(current_scaler),
                    key="preprocess_scaler",
                )

                st.session_state.use_pca = st.checkbox(
                    "Use PCA",
                    value=st.session_state.use_pca,
                    key="use_pca_check",
                )

                if st.session_state.use_pca:
                    max_components = min(10, max(1, len(st.session_state.selected_features)))
                    st.session_state.pca_components = st.slider(
                        "PCA Components",
                        1,
                        max_components,
                        min(st.session_state.pca_components, max_components),
                        key="pca_slider",
                    )

            with right:
                if st.session_state.problem_choice in ["Classification", "Regression"]:
                    st.session_state.test_size = st.slider(
                        "Test Split",
                        0.1,
                        0.4,
                        float(st.session_state.test_size),
                        0.05,
                        key="test_size_slider",
                    )

            st.markdown(
                '<div class="info-box">Encoding is applied before training. Scaling and PCA are applied after feature selection.</div>',
                unsafe_allow_html=True,
            )

    elif active_page == "Train":
        st.markdown("## Step 7 — Model Training Process")

        if encoded_df is None:
            st.markdown('<div class="warn-box">Upload a dataset first.</div>', unsafe_allow_html=True)
        elif len(st.session_state.selected_features) == 0:
            st.markdown('<div class="warn-box">Select at least one feature.</div>', unsafe_allow_html=True)
        else:
            problem = st.session_state.problem_choice

            if problem == "Classification":
                st.session_state.algo_cls = st.selectbox(
                    "Choose classification algorithm",
                    CLASSIFICATION_ALGOS,
                    index=CLASSIFICATION_ALGOS.index(st.session_state.algo_cls),
                    key="train_cls_algo",
                )
                st.markdown(
                    f'<div class="info-box">{ALGO_INFO[st.session_state.algo_cls]}</div>',
                    unsafe_allow_html=True,
                )

            elif problem == "Regression":
                st.session_state.algo_reg = st.selectbox(
                    "Choose regression algorithm",
                    REGRESSION_ALGOS,
                    index=REGRESSION_ALGOS.index(st.session_state.algo_reg),
                    key="train_reg_algo",
                )
                st.markdown(
                    f'<div class="info-box">{ALGO_INFO[st.session_state.algo_reg]}</div>',
                    unsafe_allow_html=True,
                )

            else:
                st.session_state.algo_clu = st.selectbox(
                    "Choose clustering algorithm",
                    CLUSTERING_ALGOS,
                    index=CLUSTERING_ALGOS.index(st.session_state.algo_clu),
                    key="train_clu_algo",
                )
                st.markdown(
                    f'<div class="info-box">{ALGO_INFO[st.session_state.algo_clu]}</div>',
                    unsafe_allow_html=True,
                )

                if st.session_state.algo_clu == "K-Means":
                    st.session_state.k_clusters = st.slider(
                        "Number of clusters",
                        2,
                        10,
                        st.session_state.k_clusters,
                        key="kmeans_clusters",
                    )
                elif st.session_state.algo_clu == "DBSCAN":
                    st.session_state.dbscan_eps = st.slider(
                        "EPS",
                        0.1,
                        5.0,
                        float(st.session_state.dbscan_eps),
                        0.1,
                        key="dbscan_eps",
                    )
                    st.session_state.dbscan_min_samples = st.slider(
                        "Min Samples",
                        1,
                        20,
                        int(st.session_state.dbscan_min_samples),
                        key="dbscan_min_samples",
                    )

            if st.button("Run Training", key="run_training_button"):
                st.session_state.trained = False
                st.session_state.results = {}
                st.session_state.model_bundle = None
                st.session_state.rare_class_warning = ""

                training_df = encoded_df.copy() if st.session_state.encoding_enabled else original_df.copy()
                X = training_df[st.session_state.selected_features].copy()

                if problem in ["Classification", "Regression"]:
                    target_col = st.session_state.target_col
                    y_raw = training_df[target_col].copy()

                    if problem == "Classification":
                        y, reverse_mapping = encode_class_target(y_raw)
                    else:
                        y = pd.to_numeric(y_raw, errors="coerce")
                        if pd.isna(y).sum() > 0:
                            st.error("Regression target must be numeric.")
                            st.stop()
                        y = y.values
                        reverse_mapping = None

                    X_final, imputer, scaler, pca_model = preprocess_fit(
                        X,
                        scaler_choice=st.session_state.scaler_choice,
                        use_pca=st.session_state.use_pca,
                        pca_components=st.session_state.pca_components,
                    )

                    stratify_y = None
                    if problem == "Classification":
                        stratify_y, warning_text = safe_stratify_target(y)
                        st.session_state.rare_class_warning = warning_text

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_final,
                        y,
                        test_size=st.session_state.test_size,
                        random_state=42,
                        stratify=stratify_y if problem == "Classification" else None,
                    )

                    model = (
                        build_classification_model(st.session_state.algo_cls)
                        if problem == "Classification"
                        else build_regression_model(st.session_state.algo_reg)
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    st.session_state.model_bundle = {
                        "model": model,
                        "imputer": imputer,
                        "scaler": scaler,
                        "pca_model": pca_model,
                        "features": st.session_state.selected_features,
                        "problem": problem,
                        "reverse_map": reverse_mapping,
                    }

                    results = {
                        "problem": problem,
                        "y_test": y_test,
                        "preds": preds,
                        "reverse_map": reverse_mapping,
                    }

                    if problem == "Classification":
                        results["accuracy"] = accuracy_score(y_test, preds)
                    else:
                        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                        mae = mean_absolute_error(y_test, preds)
                        safe_y = np.where(np.abs(y_test) < 1e-8, 1e-8, y_test)
                        mape = np.mean(np.abs((y_test - preds) / safe_y)) * 100
                        custom_acc = max(0, 100 - mape)
                        results["rmse"] = rmse
                        results["mae"] = mae
                        results["custom_accuracy"] = custom_acc

                    st.session_state.results = results
                    st.session_state.trained = True
                    st.success("Training completed successfully.")

                else:
                    X_final, _, _, _ = preprocess_fit(
                        X,
                        scaler_choice=st.session_state.scaler_choice,
                        use_pca=True,
                        pca_components=max(2, st.session_state.pca_components),
                    )

                    if st.session_state.algo_clu == "K-Means":
                        model = KMeans(
                            n_clusters=st.session_state.k_clusters,
                            random_state=42,
                            n_init=10,
                        )
                        labels = model.fit_predict(X_final)
                    elif st.session_state.algo_clu == "DBSCAN":
                        model = DBSCAN(
                            eps=st.session_state.dbscan_eps,
                            min_samples=st.session_state.dbscan_min_samples,
                        )
                        labels = model.fit_predict(X_final)
                    else:
                        model = AgglomerativeClustering(n_clusters=st.session_state.k_clusters)
                        labels = model.fit_predict(X_final)

                    cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
                    silhouette = None
                    if len(set(labels)) > 1 and len(X_final) > len(set(labels)):
                        try:
                            silhouette = silhouette_score(X_final, labels)
                        except Exception:
                            silhouette = None

                    st.session_state.results = {
                        "problem": "Clustering",
                        "labels": labels,
                        "X_final": X_final,
                        "cluster_count": cluster_count,
                        "silhouette": silhouette,
                    }
                    st.session_state.trained = True
                    st.success("Clustering completed successfully.")

    elif active_page == "Results":
        st.markdown("## Step 8 — Results and Output Process")

        if not st.session_state.trained or not st.session_state.results:
            st.markdown(
                '<div class="warn-box">Train a model first to view the result page.</div>',
                unsafe_allow_html=True,
            )
        else:
            res = st.session_state.results
            problem = res["problem"]

            if st.session_state.rare_class_warning:
                st.markdown(
                    f'<div class="warn-box">{st.session_state.rare_class_warning}</div>',
                    unsafe_allow_html=True,
                )

            if problem == "Classification":
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Problem", "Classification")
                r2.metric("Algorithm", st.session_state.algo_cls)
                r3.metric("Accuracy", f"{res['accuracy']:.4f}")
                r4.metric("Test Samples", len(res["y_test"]))

                p1, p2 = st.columns(2)
                with p1:
                    st.markdown("### Actual vs Predicted")
                    show_plot(plot_actual_vs_pred(res["y_test"], res["preds"]))

                with p2:
                    st.markdown("### Prediction Distribution")
                    show_plot(plot_class_distribution(res["preds"], reverse_mapping=res["reverse_map"]))

            elif problem == "Regression":
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Problem", "Regression")
                r2.metric("Algorithm", st.session_state.algo_reg)
                r3.metric("RMSE", f"{res['rmse']:.4f}")
                r4.metric("Custom Accuracy", f"{res['custom_accuracy']:.2f}%")

                p1, p2 = st.columns(2)
                with p1:
                    st.markdown("### Actual vs Predicted")
                    show_plot(plot_actual_vs_pred(res["y_test"], res["preds"]))

                with p2:
                    st.markdown("### Prediction Distribution")
                    show_plot(plot_regression_distribution(res["preds"]))

            else:
                sil_text = "N/A" if res["silhouette"] is None else f"{res['silhouette']:.4f}"
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Problem", "Clustering")
                r2.metric("Algorithm", st.session_state.algo_clu)
                r3.metric("Clusters", res["cluster_count"])
                r4.metric("Silhouette", sil_text)

                st.markdown("### Cluster Visualization")
                show_plot(plot_clusters(res["X_final"], res["labels"], f"{st.session_state.algo_clu} Clusters"))

            if st.session_state.model_bundle is not None and problem in ["Classification", "Regression"]:
                st.divider()
                st.markdown("### Manual Prediction")

                bundle = st.session_state.model_bundle
                feature_inputs = {}
                cols = st.columns(2)

                for i, feat in enumerate(bundle["features"]):
                    with cols[i % 2]:
                        default_val = (
                            float(encoded_df[feat].median())
                            if feat in encoded_df.columns and pd.api.types.is_numeric_dtype(encoded_df[feat])
                            else 0.0
                        )
                        feature_inputs[feat] = st.number_input(
                            f"Enter {feat}",
                            value=default_val,
                            key=f"manual_input_{feat}",
                        )

                if st.button("Predict Manual Input", key="predict_manual_button"):
                    input_df = pd.DataFrame([feature_inputs])
                    transformed = transform_manual_input(
                        input_df,
                        bundle["imputer"],
                        bundle["scaler"],
                        bundle["pca_model"],
                    )
                    pred = bundle["model"].predict(transformed)[0]

                    if bundle["problem"] == "Classification":
                        pred_label = bundle["reverse_map"].get(int(pred), str(pred))
                        st.markdown(
                            f'<div class="success-box">Manual Prediction Output: {pred_label}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="success-box">Manual Prediction Output: {float(pred):.4f}</div>',
                            unsafe_allow_html=True,
                        )

    elif active_page == "Visualization":
        st.markdown("## Step 9 — Visualization Process")

        if encoded_df is None:
            st.markdown('<div class="warn-box">Upload a dataset first.</div>', unsafe_allow_html=True)
        else:
            all_columns = encoded_df.columns.tolist()
            numeric_columns = encoded_df.select_dtypes(include=[np.number]).columns.tolist()

            st.markdown(
                '<div class="info-box">All visualizations support manual X and Y column selection.</div>',
                unsafe_allow_html=True,
            )

            vtabs = st.tabs(["Line", "Bar", "Scatter", "Box", "Violin", "Area"])

            with vtabs[0]:
                st.markdown("### Line Chart")
                l1, l2 = st.columns(2)
                with l1:
                    st.session_state.viz_choices["line_x"] = st.selectbox(
                        "X Column",
                        all_columns,
                        index=0 if st.session_state.viz_choices["line_x"] not in all_columns else all_columns.index(st.session_state.viz_choices["line_x"]),
                        key="line_x_col",
                    )
                with l2:
                    line_y_options = numeric_columns if numeric_columns else all_columns
                    st.session_state.viz_choices["line_y"] = st.selectbox(
                        "Y Column",
                        line_y_options,
                        index=0 if st.session_state.viz_choices["line_y"] not in line_y_options else line_y_options.index(st.session_state.viz_choices["line_y"]),
                        key="line_y_col",
                    )
                show_plot(plot_line_chart(encoded_df, st.session_state.viz_choices["line_x"], st.session_state.viz_choices["line_y"]))

            with vtabs[1]:
                st.markdown("### Bar Chart")
                b1, b2 = st.columns(2)
                with b1:
                    st.session_state.viz_choices["bar_x"] = st.selectbox(
                        "X Column",
                        all_columns,
                        index=0 if st.session_state.viz_choices["bar_x"] not in all_columns else all_columns.index(st.session_state.viz_choices["bar_x"]),
                        key="bar_x_col",
                    )
                with b2:
                    bar_y_options = numeric_columns if numeric_columns else all_columns
                    st.session_state.viz_choices["bar_y"] = st.selectbox(
                        "Y Column",
                        bar_y_options,
                        index=0 if st.session_state.viz_choices["bar_y"] not in bar_y_options else bar_y_options.index(st.session_state.viz_choices["bar_y"]),
                        key="bar_y_col",
                    )
                show_plot(plot_bar_chart(encoded_df, st.session_state.viz_choices["bar_x"], st.session_state.viz_choices["bar_y"]))

            with vtabs[2]:
                st.markdown("### Scatter Plot")
                s1, s2 = st.columns(2)
                with s1:
                    scatter_x_options = numeric_columns if numeric_columns else all_columns
                    st.session_state.viz_choices["scatter_x"] = st.selectbox(
                        "X Column",
                        scatter_x_options,
                        index=0 if st.session_state.viz_choices["scatter_x"] not in scatter_x_options else scatter_x_options.index(st.session_state.viz_choices["scatter_x"]),
                        key="scatter_x_col",
                    )
                with s2:
                    scatter_y_options = numeric_columns if numeric_columns else all_columns
                    st.session_state.viz_choices["scatter_y"] = st.selectbox(
                        "Y Column",
                        scatter_y_options,
                        index=0 if st.session_state.viz_choices["scatter_y"] not in scatter_y_options else scatter_y_options.index(st.session_state.viz_choices["scatter_y"]),
                        key="scatter_y_col",
                    )
                show_plot(plot_scatter_chart(encoded_df, st.session_state.viz_choices["scatter_x"], st.session_state.viz_choices["scatter_y"]))

            with vtabs[3]:
                st.markdown("### Box Plot")
                x1, x2 = st.columns(2)
                with x1:
                    st.session_state.viz_choices["box_x"] = st.selectbox(
                        "Group Column (X)",
                        all_columns,
                        index=0 if st.session_state.viz_choices["box_x"] not in all_columns else all_columns.index(st.session_state.viz_choices["box_x"]),
                        key="box_x_col",
                    )
                with x2:
                    box_y_options = numeric_columns if numeric_columns else all_columns
                    st.session_state.viz_choices["box_y"] = st.selectbox(
                        "Value Column (Y)",
                        box_y_options,
                        index=0 if st.session_state.viz_choices["box_y"] not in box_y_options else box_y_options.index(st.session_state.viz_choices["box_y"]),
                        key="box_y_col",
                    )
                show_plot(plot_box_chart(encoded_df, st.session_state.viz_choices["box_x"], st.session_state.viz_choices["box_y"]))

            with vtabs[4]:
                st.markdown("### Violin Plot")
                v1, v2 = st.columns(2)
                with v1:
                    st.session_state.viz_choices["violin_x"] = st.selectbox(
                        "Group Column (X)",
                        all_columns,
                        index=0 if st.session_state.viz_choices["violin_x"] not in all_columns else all_columns.index(st.session_state.viz_choices["violin_x"]),
                        key="violin_x_col",
                    )
                with v2:
                    violin_y_options = numeric_columns if numeric_columns else all_columns
                    st.session_state.viz_choices["violin_y"] = st.selectbox(
                        "Value Column (Y)",
                        violin_y_options,
                        index=0 if st.session_state.viz_choices["violin_y"] not in violin_y_options else violin_y_options.index(st.session_state.viz_choices["violin_y"]),
                        key="violin_y_col",
                    )
                show_plot(plot_violin_chart(encoded_df, st.session_state.viz_choices["violin_x"], st.session_state.viz_choices["violin_y"]))

            with vtabs[5]:
                st.markdown("### Area Chart")
                a1, a2 = st.columns(2)
                with a1:
                    st.session_state.viz_choices["area_x"] = st.selectbox(
                        "X Column",
                        all_columns,
                        index=0 if st.session_state.viz_choices["area_x"] not in all_columns else all_columns.index(st.session_state.viz_choices["area_x"]),
                        key="area_x_col",
                    )
                with a2:
                    area_y_options = numeric_columns if numeric_columns else all_columns
                    st.session_state.viz_choices["area_y"] = st.selectbox(
                        "Y Column",
                        area_y_options,
                        index=0 if st.session_state.viz_choices["area_y"] not in area_y_options else area_y_options.index(st.session_state.viz_choices["area_y"]),
                        key="area_y_col",
                    )
                show_plot(plot_area_chart(encoded_df, st.session_state.viz_choices["area_x"], st.session_state.viz_choices["area_y"]))