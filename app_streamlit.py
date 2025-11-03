import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

# =============== Helpers ===============

@st.cache_data(show_spinner=False)
def generate_synthetic_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    study_hours = rng.normal(2.5, 1.0, n).clip(0, 6)
    attendance_rate = rng.normal(0.85, 0.08, n).clip(0.4, 1.0)
    past_avg = rng.normal(65, 12, n).clip(0, 100)
    quiz_mean = rng.normal(6.8, 1.4, n).clip(0, 10)
    assignments_done = rng.integers(0, 12, n)
    difficulty = rng.choice(["baixa", "média", "alta"], size=n, p=[0.35, 0.45, 0.20])
    support = rng.choice(["nenhum", "escola", "casa", "ambos"], size=n, p=[0.25, 0.35, 0.25, 0.15])

    diff_map = {"baixa": +4, "média": 0, "alta": -6}
    supp_map = {"nenhum": -2, "escola": +2, "casa": +2, "ambos": +4}
    noise = rng.normal(0, 5.0, n)

    final_grade = (
        10*study_hours +
        25*attendance_rate +
        0.4*past_avg +
        3.0*quiz_mean +
        1.2*assignments_done +
        np.vectorize(diff_map.get)(difficulty) +
        np.vectorize(supp_map.get)(support) +
        noise
    ).clip(0, 100)

    passed = (final_grade >= 60).astype(int)

    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance_rate": attendance_rate,
        "past_avg": past_avg,
        "quiz_mean": quiz_mean,
        "assignments_done": assignments_done,
        "difficulty": difficulty,
        "support": support,
        "final_grade": final_grade.round(1),
        "passed": passed
    })
    return df

def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

@st.cache_resource(show_spinner=False)
def train_models(df, n_estimators=300, seed=42):
    features = ["study_hours","attendance_rate","past_avg","quiz_mean","assignments_done","difficulty","support"]
    num = ["study_hours","attendance_rate","past_avg","quiz_mean","assignments_done"]
    cat = ["difficulty","support"]

    X = df[features]
    y_reg = df["final_grade"]
    y_clf = df["passed"]

    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y_reg, test_size=0.2, random_state=seed)
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X, y_clf, test_size=0.2, random_state=seed, stratify=y_clf)

    preprocess = build_preprocessor(num, cat)

    reg = Pipeline([("prep", preprocess),
                    ("rf", RandomForestRegressor(n_estimators=n_estimators, random_state=seed))])
    clf = Pipeline([("prep", preprocess),
                    ("rf", RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=seed))])

    reg.fit(Xtr_r, ytr_r)
    clf.fit(Xtr_c, ytr_c)

    # métricas rápidas
    pred_r = reg.predict(Xte_r)
    mae = mean_absolute_error(yte_r, pred_r)
    r2  = r2_score(yte_r, pred_r)

    pred_c = clf.predict(Xte_c)
    acc = accuracy_score(yte_c, pred_c)
    f1  = f1_score(yte_c, pred_c)

    metrics = {
        "MAE (regressão)": round(mae, 2),
        "R² (regressão)": round(r2, 3),
        "Acurácia (classificação)": round(acc, 3),
        "F1 (classificação)": round(f1, 3)
    }

    return reg, clf, metrics, features

def ensure_required_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    return missing

def build_template(features):
    # gera um template com 3 linhas de exemplo
    tmp = pd.DataFrame([{
        "study_hours": 2.5,
        "attendance_rate": 0.90,
        "past_avg": 70,
        "quiz_mean": 7.0,
        "assignments_done": 8,
        "difficulty": "média",
        "support": "escola"
    },{
        "study_hours": 1.0,
        "attendance_rate": 0.75,
        "past_avg": 55,
        "quiz_mean": 6.0,
        "assignments_done": 5,
        "difficulty": "alta",
        "support": "nenhum"
    },{
        "study_hours": 3.5,
        "attendance_rate": 0.95,
        "past_avg": 85,
        "quiz_mean": 8.5,
        "assignments_done": 10,
        "difficulty": "baixa",
        "support": "ambos"
    }])
    return tmp[features]

# =============== UI ===============

st.set_page_config(page_title="Previsor de Notas", page_icon="🎓", layout="wide")
st.title(" Previsor de Notas / Desempenho de Alunos")

with st.sidebar:
    st.header("Configurações")
    seed = st.number_input("Seed (reprodutibilidade)", min_value=0, value=42, step=1)
    n_data = st.slider("Tamanho do conjunto sintético (n)", 200, 3000, 800, step=100)
    n_trees = st.slider("Nº de árvores (RandomForest)", 50, 800, 300, step=50)
    st.caption("Valores maiores tendem a melhorar, mas deixam o treino mais lento.")

df = generate_synthetic_data(n=n_data, seed=seed)
reg_model, clf_model, metrics, features = train_models(df, n_estimators=n_trees, seed=seed)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔹 Previsão individual",
    "📦 Previsão em lote (CSV)",
    "📈 Métricas do modelo",
    "🗃️ Dados de exemplo"
])

with tab1:
    st.subheader("Entradas do aluno")
    c1, c2, c3 = st.columns(3)
    with c1:
        study_hours = st.slider("Horas de estudo/dia", 0.0, 6.0, 2.5, 0.1)
        past_avg = st.slider("Média passada (0–100)", 0, 100, 65, 1)
    with c2:
        attendance_rate = st.slider("Frequência (%)", 40, 100, 85, 1) / 100.0
        quiz_mean = st.slider("Média dos quizzes (0–10)", 0.0, 10.0, 6.5, 0.1)
    with c3:
        assignments_done = st.slider("Tarefas feitas (0–11)", 0, 11, 8, 1)
        difficulty = st.selectbox("Dificuldade", ["baixa","média","alta"])
        support = st.selectbox("Suporte", ["nenhum","escola","casa","ambos"])

    input_df = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_rate": attendance_rate,
        "past_avg": past_avg,
        "quiz_mean": quiz_mean,
        "assignments_done": assignments_done,
        "difficulty": difficulty,
        "support": support
    }])

    st.markdown("### Resultado")
    mode = st.radio("O que deseja prever?", ["Nota (0–100)", "Aprovação (Sim/Não)"], horizontal=True)

    if mode == "Nota (0–100)":
        pred = reg_model.predict(input_df)[0]
        st.metric("Nota prevista", f"{pred:.1f} / 100")
    else:
        thr = st.slider("Limiar de aprovação (0–1)", 0.1, 0.9, 0.5, 0.05,
                        help="Se probabilidade ≥ limiar, o aluno é considerado Aprovado.")
        proba = clf_model.predict_proba(input_df)[0,1]
        label = "✅ Aprovado" if proba >= thr else "❌ Reprovado"
        st.metric("Probabilidade de aprovação", f"{proba*100:.1f}%")
        st.write(label)

    with st.expander("Ver entrada usada"):
        st.dataframe(input_df, use_container_width=True)

with tab2:
    st.subheader("Suba um CSV e gere previsões para vários alunos")
    st.write("O CSV precisa ter **exatamente** estas colunas:")
    st.code(", ".join(features))

    uploaded = st.file_uploader("Envie seu arquivo .csv", type=["csv"])
    template = build_template(features)
    st.download_button("Baixar CSV de exemplo", template.to_csv(index=False).encode("utf-8"),
                       file_name="template_alunos.csv", mime="text/csv")

    if uploaded:
        try:
            df_u = pd.read_csv(uploaded)
            missing = ensure_required_columns(df_u, features)
            if missing:
                st.error(f"Seu CSV está faltando colunas: {missing}")
            else:
                st.success("CSV válido! Gerando previsões…")
                # Faz as duas previsões
                y_reg_pred = reg_model.predict(df_u[features])
                y_clf_prob = clf_model.predict_proba(df_u[features])[:,1]
                out = df_u.copy()
                out["pred_final_grade"] = np.round(y_reg_pred, 1)
                out["prob_pass"] = np.round(y_clf_prob, 4)
                st.dataframe(out.head(50), use_container_width=True)
                st.download_button("Baixar resultados (CSV)",
                                   out.to_csv(index=False).encode("utf-8"),
                                   file_name="previsoes_alunos.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Erro ao ler/processar o CSV: {e}")

with tab3:
    st.subheader("Métricas rápidas (com separação treino/teste interna)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (Regressão)", metrics["MAE (regressão)"])
    c2.metric("R² (Regressão)", metrics["R² (regressão)"])
    c3.metric("Acurácia (Class.)", metrics["Acurácia (classificação)"])
    c4.metric("F1 (Class.)", metrics["F1 (classificação)"])
    st.caption("OBS: as métricas usam dados SINTÉTICOS e servem apenas como referência didática.")

with tab4:
    st.subheader("Amostra dos dados sintéticos")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption("Você pode aumentar 'n' na barra lateral para gerar mais exemplos.")
