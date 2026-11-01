import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Wine Quality — оценка вина", page_icon="🍷", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(APP_DIR / "test.csv")


@st.cache_data
def load_config():
    return json.loads((APP_DIR / "config.json").read_text(encoding="utf-8"))


try:
    model = load_model()
    test_df = load_data()
    config = load_config()
except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
    st.error(f"Не удалось загрузить файлы приложения: {exc}")
    st.stop()

feature_columns = list(test_df.drop(columns="target").columns)
if len(feature_columns) != 12 or "is_red" not in feature_columns:
    st.error("В test.csv ожидается 12 признаков, включая is_red, и столбец target.")
    st.stop()

if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    if set(model_features) != set(feature_columns):
        st.error("Признаки в test.csv не совпадают с признаками обученной модели.")
        st.stop()
    feature_columns = model_features

try:
    default_threshold = float(config["threshold"])
except (KeyError, TypeError, ValueError) as exc:
    st.error(f"Некорректный порог в config.json: {exc}")
    st.stop()

if not 0 <= default_threshold <= 1:
    st.error("Порог в config.json должен быть в диапазоне [0, 1].")
    st.stop()

st.title("Оценка качества вина")
st.caption("Оценка вероятности качества ≥ 7 по результатам лабораторного анализа")

with st.sidebar:
    st.header("Настройки подборки")
    threshold = st.slider(
        "Порог отбора",
        min_value=0.0,
        max_value=1.0,
        value=default_threshold,
        step=0.001,
        format="%.3f",
        help="Вино рекомендуется, если вероятность хорошего качества не ниже порога.",
    )
    st.caption(f"Порог из config.json: {default_threshold:.3f}")

tab_wine, tab_metrics, tab_batch, tab_coef = st.tabs([
    "Оценка вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"
])

# st.cache_resource хранит один общий объект модели (включая её состояние).
# st.cache_data хранит результаты вычислений/таблицы и возвращает их копии.
# Без кэша joblib.load читал бы файл заново при каждом изменении виджетов.
@st.cache_data(show_spinner=False)
def get_test_scores(_model, data, columns):
    X = data.loc[:, list(columns)]
    y = data["target"].to_numpy(dtype=int)
    scores = _model.predict_proba(X)[:, 1]
    return y, scores


with tab_wine:
    st.subheader("Характеристики вина")
    st.write("Введите измеренные характеристики. Значения по умолчанию — медианы тестовой выборки.")

    values = {}
    left, right = st.columns(2, gap="large")
    numeric_features = [col for col in feature_columns if col != "is_red"]

    for idx, feature in enumerate(numeric_features):
        series = pd.to_numeric(test_df[feature], errors="coerce").dropna()
        if series.empty:
            st.error(f"Нет числовых значений для признака {feature}.")
            st.stop()

        min_value = float(series.min())
        max_value = float(series.max())
        median_value = float(series.median())
        target_column = left if idx % 2 == 0 else right

        with target_column:
            values[feature] = st.number_input(
                feature.replace("_", " ").capitalize(),
                min_value=min_value,
                max_value=max_value,
                value=median_value,
                step=max((max_value - min_value) / 1000, 0.00001),
                format="%.5f",
                key=f"feature_{feature}",
                help=f"Диапазон в тестовой выборке: [{min_value:g}, {max_value:g}]",
            )

    values["is_red"] = int(st.toggle("Красное вино (выключено — белое)", value=False))

    # Сохраняем порядок признаков, использованный при обучении модели.
    wine = pd.DataFrame([{name: values[name] for name in feature_columns}], columns=feature_columns)

    probability = float(model.predict_proba(wine)[0, 1])
    recommended = probability >= threshold

    st.divider()
    st.subheader("Результат оценки")
    score_col, threshold_col = st.columns(2)
    score_col.metric("Вероятность хорошего вина", f"{probability:.1%}")
    threshold_col.metric("Порог отбора", f"{threshold:.1%}")

    st.progress(min(max(probability, 0.0), 1.0))
    if recommended:
        st.success("Рекомендовать: вино проходит установленный порог.")
    else:
        st.warning("Не рекомендовать: вероятность ниже установленного порога.")

    st.caption("Вероятность рассчитана обученной моделью. Изменение порога влияет только на решение о рекомендации.")


with tab_metrics:
    st.subheader("Качество модели на тестовой выборке")
    st.caption(
        "Метрики рассчитываются на неизменной тестовой выборке. "
        "При изменении порога меняются предсказанные классы, но не оценки модели."
    )

    # Кэшируем вероятности на тесте: порог не является аргументом функции.
    y_true, scores = get_test_scores(model, test_df, tuple(feature_columns))
    y_pred = (scores >= threshold).astype(int)

    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

    col_p, col_r, col_f = st.columns(3)
    col_p.metric("Precision", f"{precision:.3f}")
    col_r.metric("Recall", f"{recall:.3f}")
    col_f.metric("F1", f"{f1:.3f}")

    st.subheader("Матрица ошибок")
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
    disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Не хорошее (0)", "Хорошее (1)"],
    )
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False, values_format="d")
    ax_cm.grid(False)
    for label in disp.text_.ravel():
        label.set_fontsize(15)
    fig_cm.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    @st.cache_data(show_spinner=False)
    def get_curves(y, proba):
        fpr, tpr, _ = metrics.roc_curve(y, proba)
        pr_precision, pr_recall, _ = metrics.precision_recall_curve(y, proba)
        return (fpr, tpr, pr_precision, pr_recall,
                metrics.roc_auc_score(y, proba),
                metrics.average_precision_score(y, proba))

    fpr, tpr, pr_precision, pr_recall, roc_auc, ap = get_curves(y_true, scores)

    # Выделенная точка считается при том же условии >=, что и решение модели.
    tn, fp, fn, tp = cm.ravel()
    fpr_point = fp / (fp + tn) if fp + tn else 0.0
    tpr_point = recall

    st.subheader("ROC- и PR-кривые")
    roc_col, pr_col = st.columns(2, gap="large")

    with roc_col:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
        ax_roc.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {roc_auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], "k:", label="Случайная модель")
        ax_roc.scatter(fpr_point, tpr_point, color="crimson", s=85,
                       edgecolors="white", linewidths=1.2, zorder=5,
                       label=f"Порог = {threshold:.3f}")
        ax_roc.set(xlabel="FPR", ylabel="TPR (Recall)",
                   title="ROC", xlim=(0, 1), ylim=(0, 1.02))
        ax_roc.grid(alpha=0.2)
        ax_roc.legend(loc="lower right")
        fig_roc.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    with pr_col:
        fig_pr, ax_pr = plt.subplots(figsize=(6, 4.5))
        ax_pr.plot(pr_recall, pr_precision, linewidth=2,
                   label=f"Average Precision = {ap:.3f}")
        ax_pr.axhline(y_true.mean(), color="black", linestyle=":",
                      label=f"Случайная модель = {y_true.mean():.3f}")
        ax_pr.scatter(recall, precision, color="crimson", s=85,
                      edgecolors="white", linewidths=1.2, zorder=5,
                      label=f"Порог = {threshold:.3f}")
        ax_pr.set(xlabel="Recall", ylabel="Precision",
                  title="Precision–Recall", xlim=(0, 1), ylim=(0, 1.02))
        ax_pr.grid(alpha=0.2)
        ax_pr.legend(loc="upper right")
        fig_pr.tight_layout()
        st.pyplot(fig_pr)
        plt.close(fig_pr)

    st.info(
        "Для сомелье: при повышении порога подборка становится меньше — "
        "как правило, в ней повышается доля действительно хороших вин "
        "(precision), но больше хороших вин пропускается (recall падает). "
        "При снижении порога рекомендаций становится больше: "
        "recall растёт, однако увеличивается риск рекомендовать посредственное вино. "
        "Изменения precision не обязаны быть монотонными."
    )

with tab_batch:
    st.subheader("Пакетное предсказание по CSV")
    st.write("Загрузите CSV с 12 признаками вина без `target`. Используется общий порог из боковой панели.")
    uploaded_file = st.file_uploader("CSV-файл с винами", type="csv", key="batch_csv")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except (pd.errors.ParserError, UnicodeError, ValueError, pd.errors.EmptyDataError) as exc:
            st.error(f"Не удалось прочитать CSV: {exc}")
        else:
            missing = [col for col in feature_columns if col not in batch_df.columns]
            if missing:
                st.error("В файле отсутствуют столбцы: " + ", ".join(missing))
            elif "target" in batch_df.columns:
                st.error("Удалите столбец `target`: файл должен содержать только признаки.")
            elif batch_df.empty:
                st.error("CSV не содержит ни одного вина.")
            else:
                batch_features = batch_df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
                invalid = batch_features.isna() | ~np.isfinite(batch_features)
                if invalid.any().any():
                    bad_cols = invalid.columns[invalid.any()].tolist()
                    st.error("Пропущенные или некорректные значения в столбцах: " + ", ".join(bad_cols))
                elif not batch_features["is_red"].isin([0, 1]).all():
                    st.error("`is_red` должен содержать только 0 (белое) или 1 (красное).")
                else:
                    probabilities = model.predict_proba(batch_features)[:, 1]
                    decisions = (probabilities >= threshold).astype(int)
                    predictions_df = batch_features.copy()
                    predictions_df["probability_good"] = probabilities
                    predictions_df["recommended"] = decisions

                    st.metric("Вин в подборке", f"{decisions.sum()} из {len(decisions)}")
                    st.caption(f"Порог: {threshold:.3f}. recommended = 1 — рекомендовать.")
                    st.dataframe(
                        predictions_df.style.format({"probability_good": "{:.4f}"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.download_button(
                        "Скачать результаты CSV",
                        data=predictions_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="wine_predictions.csv",
                        mime="text/csv",
                    )


with tab_coef:
    st.subheader("Коэффициенты логистической регрессии")
    st.caption(
        "Коэффициенты получены на стандартизированных признаках: "
        "положительные значения повышают логарифм отношения шансов хорошего вина, "
        "отрицательные — понижают. Остальные признаки считаются фиксированными."
    )

    # Сохраняем порядок этапов обученного imblearn Pipeline.
    if not hasattr(model, "steps"):
        st.error("Сохранённая модель не является пайплайном.")
        st.stop()

    regressor = model.steps[-1][1]
    if not hasattr(regressor, "coef_") or not hasattr(regressor, "classes_"):
        st.error("Последний этап модели не содержит коэффициентов классификатора.")
        st.stop()

    classes = list(regressor.classes_)
    if 1 not in classes or len(classes) != 2:
        st.error("Для визуализации ожидается бинарная логистическая регрессия с классами 0 и 1.")
        st.stop()

    coef = np.asarray(regressor.coef_)
    if coef.shape != (1, len(feature_columns)):
        st.error("Размерность коэффициентов не совпадает с числом признаков.")
        st.stop()

    # При классе 1 в classes_[1] положительные коэффициенты голосуют за 1.
    signed_coef = coef[0] if classes[1] == 1 else -coef[0]
    coefficients = pd.Series(signed_coef, index=feature_columns).sort_values()

    fig_coef, ax_coef = plt.subplots(figsize=(9, 6))
    colors = ["#c25252" if value < 0 else "#279279" for value in coefficients]
    ax_coef.barh(coefficients.index, coefficients.values, color=colors)
    ax_coef.axvline(0, color="black", linewidth=1)
    ax_coef.set(
        xlabel="Коэффициент (после StandardScaler)",
        ylabel="Признак",
        title="Влияние признаков на логарифм шансов класса 1",
    )
    ax_coef.grid(axis="x", alpha=0.2)
    ax_coef.set_axisbelow(True)
    fig_coef.tight_layout()
    st.pyplot(fig_coef)
    plt.close(fig_coef)

    positive = coefficients[coefficients > 0].sort_values(ascending=False)
    negative = coefficients[coefficients < 0].sort_values()
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**Сильнее всего за хорошее вино**")
        if positive.empty:
            st.write("Нет положительных коэффициентов")
        else:
            st.write(", ".join(f"{name} ({value:+.3f})" for name, value in positive.head(3).items()))
    with col_neg:
        st.markdown("**Сильнее всего против хорошего вина**")
        if negative.empty:
            st.write("Нет отрицательных коэффициентов")
        else:
            st.write(", ".join(f"{name} ({value:+.3f})" for name, value in negative.head(3).items()))

    st.caption(
        "Это ассоциации модели, не доказательство причинного влияния. "
        "Для бинарного is_red изменение на единицу не равно одному стандартному "
        "отклонению исходного показателя. При коррелированных признаках "
        "сравнивать коэффициенты следует осторожно."
    )
