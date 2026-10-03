# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(
    page_title="信用卡交易詐欺風控決策原型",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DATA_FILE = ROOT / "demo_data.csv"

REQUIRED = [
    ART / "scaler.pkl",
    ART / "lr_model.pkl",
    ART / "iso_model.pkl",
    ART / "xgb_model.json",
    ART / "feature_cols.json",
    ART / "thresholds.json",
    ART / "metrics.json",
    ART / "shap_importance.json",
    DATA_FILE,
]
missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
if missing:
    st.error("缺少網站所需檔案：" + ", ".join(missing))
    st.stop()

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(ART / "scaler.pkl")
    lr = joblib.load(ART / "lr_model.pkl")
    iso = joblib.load(ART / "iso_model.pkl")
    xgb = XGBClassifier()
    xgb.load_model(ART / "xgb_model.json")
    with open(ART / "feature_cols.json", encoding="utf-8") as f:
        feature_cols = json.load(f)
    with open(ART / "thresholds.json", encoding="utf-8") as f:
        thresholds = json.load(f)
    with open(ART / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(ART / "shap_importance.json", encoding="utf-8") as f:
        shap_importance = json.load(f)
    return scaler, lr, iso, xgb, feature_cols, thresholds, metrics, shap_importance

@st.cache_data
def load_demo_data():
    d = pd.read_csv(DATA_FILE)
    d = d.dropna(subset=["Class"]).copy()
    d["Class"] = d["Class"].astype(int)
    return d

scaler, lr_model, iso_model, xgb_model, feature_cols, thresholds, metrics, shap_importance = load_artifacts()
df = load_demo_data()

@st.cache_data
def score_demo(d: pd.DataFrame):
    X = d[feature_cols]
    Xs = scaler.transform(X)
    out = d.copy()
    out["xgb_prob"] = xgb_model.predict_proba(Xs)[:, 1]
    out["lr_prob"] = lr_model.predict_proba(Xs)[:, 1]
    raw_iso = -iso_model.decision_function(Xs)
    out["iso_raw"] = raw_iso
    out["iso_percentile"] = pd.Series(raw_iso).rank(pct=True).to_numpy()
    return out

df_eval = score_demo(df)
y = df_eval["Class"].to_numpy()
ds = metrics["dataset"]

def cm_at_threshold(th):
    p = (df_eval["xgb_prob"].to_numpy() >= th).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    return p, tn, fp, fn, tp

def action(prob, block_th):
    review_th = min(0.70, max(0.50, block_th - 0.10))
    if prob >= block_th:
        return "🚨 Block"
    if prob >= review_th:
        return "⚠️ Manual Review"
    if prob >= 0.50:
        return "📱 OTP / 3DS"
    return "✅ Pass"

def metric_explainer(title, value, explanation):
    st.metric(title, value)
    st.caption(explanation)

# ---------- Sidebar ----------
st.sidebar.title("🛡️ Fraud Risk Lab")
st.sidebar.caption("信用卡交易詐欺風控決策原型")

page = st.sidebar.radio(
    "功能導覽",
    [
        "🏠 風控總覽",
        "⚡ 近即時交易回放",
        "🤖 模型競技場",
        "💰 成本敏感實驗室",
        "🎚️ 風控策略模擬器",
        "🔍 XAI 模型解釋",
        "🎯 系統主要功能與應用對象",
    ],
)

st.sidebar.markdown("---")
cost_label = st.sidebar.selectbox(
    "FN : FP 成本情境",
    ["5:1", "10:1", "20:1"],
    index=1,
    help="FN 是漏掉詐欺；FP 是誤報正常交易。此權重用於模擬不同風險胃納。",
)
recommended = float(thresholds[cost_label])
st.sidebar.caption(f"Validation Set 推薦門檻：**{recommended:.3f}**")
st.sidebar.caption("完整模型離線訓練；網站使用獨立 Test Set 進行回放與決策模擬。")

# ---------- Shared current scenario ----------
pred, tn, fp, fn, tp = cm_at_threshold(recommended)
total_n = len(df_eval)
normal_n = int((y == 0).sum())
fraud_n = int(y.sum())
alert_n = int(pred.sum())
alert_rate = alert_n / total_n
fraud_rate = fraud_n / total_n
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
fp10k = fp / max(normal_n, 1) * 10000
captured_amount = float(df_eval.loc[(pred == 1) & (y == 1), "Amount"].sum())
cost_fn = int(cost_label.split(":")[0])
weighted_cost = fn * cost_fn + fp
time_min = float(df_eval["Time"].min())
time_max = float(df_eval["Time"].max())

st.title("🛡️ 信用卡交易詐欺風控決策原型")
st.caption(
    "機器學習 × 成本敏感門檻 × 可解釋 AI｜完整 284,807 筆資料離線訓練｜"
    "70% Train / 15% Validation / 15% Test 時序切分"
)

# ---------- Page 1 ----------
if page == "🏠 風控總覽":
    st.subheader("一眼看懂：目前風控系統發生什麼事？")
    st.info(
        "💡 **怎麼看？** 真實詐欺率是資料中已標記為 Fraud 的比例；警報率是模型依門檻挑出的待處理交易比例。"
        "**兩者不是同一件事。**"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_explainer("Test Set 交易數", f"{total_n:,}", "獨立測試資料，未使用 SMOTE。")
    with c2:
        metric_explainer("真實詐欺率", f"{fraud_rate*100:.3f}%", f"{fraud_n} 筆交易的真實標籤為 Fraud。")
    with c3:
        metric_explainer("模型警報率", f"{alert_rate*100:.3f}%", f"門檻 {recommended:.3f} 下，共 {alert_n} 筆被列為高風險。")
    with c4:
        metric_explainer("預估可避免損失*", f"${captured_amount:,.2f}", "離線模擬：Test Set 中被模型命中的詐欺交易 Amount 加總。")

    st.caption(
        f"* 預估可避免損失 = Σ Amount（真實 Fraud 且模型判為高風險）。"
        f"資料範圍為 Test Set 的 Time={time_min:,.0f} 至 {time_max:,.0f} 秒；這是歷史資料離線模擬，不代表銀行實際挽回金額。"
    )

    st.markdown("### 🎯 基準情境的風控結果")
    a, b, c, d, e = st.columns(5)
    a.metric("TP｜抓到詐欺", tp)
    b.metric("FP｜誤報正常", fp)
    c.metric("FN｜漏掉詐欺", fn)
    d.metric("每萬筆正常交易誤報", f"{fp10k:.2f}")
    e.metric(f"加權成本（{cost_label}）", weighted_cost)

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### 📊 風險分數分布")
        bins = pd.cut(df_eval["xgb_prob"], bins=[0, .1, .3, .5, .7, .9, 1], include_lowest=True)
        dist = pd.crosstab(bins, df_eval["Class"])
        dist.columns = ["Normal", "Fraud"]
        dist = dist.reset_index()
        dist.columns = ["風險區間", "Normal", "Fraud"]
        dist["風險區間"] = dist["風險區間"].astype(str)
        st.bar_chart(dist, x="風險區間", y=["Normal", "Fraud"])
        st.caption("XGBoost 分數越高，代表模型判定為詐欺的機率越高。")
    with right:
        st.markdown("### 🧭 銀行怎麼處理？")
        st.markdown(
            """
            **低風險 → Pass**  
            正常授權，不增加額外摩擦。

            **中度風險 → OTP / 3DS**  
            增加一次身分驗證。

            **較高風險 → Manual Review**  
            交由風控人員人工檢查。

            **超過攔截門檻 → Block**  
            暫停交易並進一步確認。
            """
        )
        st.warning("處置分級是研究原型的決策設計，不代表特定銀行實際作業規則。")

# ---------- Page 2 ----------
elif page == "⚡ 近即時交易回放":
    st.subheader("近即時交易回放：模型如何把一筆交易變成風控決策？")
    st.info("💡 **怎麼看？** XGBoost 顯示的是詐欺機率；Isolation Forest 顯示的是異常程度。兩者不能當成同一種分數。")

    s1, s2 = st.columns([1, 2])
    with s1:
        sample_size = st.slider("回放交易筆數", 5, 30, 12)
        replay_mode = st.radio("抽樣方式", ["混合抽樣", "高風險優先"], horizontal=True)
        replay_th = st.slider("本頁 Block 門檻", 0.01, 0.99, float(round(recommended, 3)), 0.001)
    if replay_mode == "高風險優先":
        sample = df_eval.nlargest(sample_size, "xgb_prob").copy()
    else:
        f = df_eval[df_eval["Class"] == 1].sample(n=min(3, fraud_n), random_state=7)
        n = df_eval[df_eval["Class"] == 0].sample(n=max(0, sample_size-len(f)), random_state=7)
        sample = pd.concat([f, n]).sample(frac=1, random_state=7).copy()

    sample["XGBoost 詐欺機率"] = (sample["xgb_prob"] * 100).round(2)
    sample["Isolation Forest 異常百分位"] = (sample["iso_percentile"] * 100).round(1)
    sample["建議處置"] = sample["xgb_prob"].map(lambda p: action(float(p), replay_th))
    sample["真實結果"] = sample["Class"].map({0: "Normal", 1: "Fraud"})
    show = sample[["Time", "Amount", "XGBoost 詐欺機率", "Isolation Forest 異常百分位", "建議處置", "真實結果"]]
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.markdown("### 📌 選一筆交易深入看")
    idx = st.selectbox("交易列", sample.index.tolist(), format_func=lambda i: f"Index {i}｜Amount ${sample.loc[i,'Amount']:.2f}")
    row = sample.loc[idx]
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("交易金額", f"${row['Amount']:,.2f}")
    q2.metric("XGBoost 詐欺機率", f"{row['xgb_prob']*100:.2f}%")
    q3.metric("Isolation 異常百分位", f"{row['iso_percentile']*100:.1f}%")
    q4.metric("決策", action(float(row["xgb_prob"]), replay_th))
    st.caption("Isolation Forest 的百分位只是為了介面閱讀方便；原始 Isolation Forest 分數不是機率。")

# ---------- Page 3 ----------
elif page == "🤖 模型競技場":
    st.subheader("三模型共同 Test Set：公平比較")
    st.info(
        f"💡 **怎麼看？** 三個模型都使用同一個、未經 SMOTE 的最終 Test Set：N={ds['test_n']:,}，"
        f"其中 Fraud={ds['test_fraud_n']}。這樣才可以公平比較。"
    )

    rows = []
    for name, m in metrics["models"].items():
        rows.append({
            "模型": name,
            "PR-AUC": m["pr_auc"],
            "ROC-AUC": m["roc_auc"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "TP": m["tp"],
            "FP": m["fp"],
            "FN": m["fn"],
            "每萬筆誤報": m["fp"] / max(normal_n, 1) * 10000,
            "門檻": m["threshold"],
        })
    bench = pd.DataFrame(rows)
    st.dataframe(
        bench.style.format({
            "PR-AUC":"{:.4f}", "ROC-AUC":"{:.4f}", "Precision":"{:.4f}",
            "Recall":"{:.4f}", "F1":"{:.4f}", "每萬筆誤報":"{:.2f}", "門檻":"{:.3f}"
        }),
        use_container_width=True, hide_index=True
    )

    chart = bench.set_index("模型")[["PR-AUC", "Precision", "Recall", "F1"]]
    st.markdown("### 📊 四個關鍵指標")
    st.bar_chart(chart)

    x = metrics["models"]["XGBoost"]
    st.markdown("### 🧠 指標翻成白話")
    a, b, c, d = st.columns(4)
    with a:
        metric_explainer("Precision", f"{x['precision']*100:.2f}%", "每 100 筆被 XGBoost 判為高風險的交易，約有多少筆真的 Fraud。")
    with b:
        metric_explainer("Recall", f"{x['recall']*100:.2f}%", "每 100 筆真正 Fraud，模型約能抓到多少筆。")
    with c:
        metric_explainer("PR-AUC", f"{x['pr_auc']:.4f}", "適合高度不平衡資料，重點看少數 Fraud 的辨識品質。")
    with d:
        metric_explainer("F1", f"{x['f1']:.4f}", "Precision 與 Recall 的綜合平衡。")

    st.warning(
        "Isolation Forest 是異常偵測模型，它的 score 不是詐欺機率；表中的門檻是該模型自身分數尺度，不能直接與 XGBoost 的 0–1 機率門檻比較。"
    )

# ---------- Page 4 ----------
elif page == "💰 成本敏感實驗室":
    st.subheader("成本敏感實驗室：為什麼 0.5 不一定是最佳答案？")
    st.info(
        "💡 **怎麼看？** FN 是漏掉 Fraud，FP 是誤報正常交易。若銀行認為 FN 更昂貴，就可能願意降低門檻，"
        "用更多 FP 換取更少 FN。門檻只在 Validation Set 決定，Test Set 只做最後驗證。"
    )

    scenario_rows = []
    for label in ["5:1", "10:1", "20:1"]:
        s = metrics["cost_scenarios"][label]
        scenario_rows.append({
            "FN:FP": label,
            "Validation 最佳門檻": s["validation_best_threshold"],
            "Test TP": s["test_tp"],
            "Test FP": s["test_fp"],
            "Test FN": s["test_fn"],
            "每萬筆正常交易誤報": s["false_positives_per_10k_normal"],
            "Test 加權成本": s["test_weighted_cost"],
        })
    sdf = pd.DataFrame(scenario_rows)
    st.dataframe(
        sdf.style.format({"Validation 最佳門檻":"{:.3f}", "每萬筆正常交易誤報":"{:.2f}"}),
        use_container_width=True, hide_index=True
    )

    st.markdown("### 🔄 成本情境變化")
    cost_chart = sdf[["FN:FP", "Test FP", "Test FN"]].copy()
    cost_chart = cost_chart.rename(columns={"FN:FP": "成本情境", "Test FP": "誤報 FP", "Test FN": "漏報 FN"})
    st.line_chart(cost_chart, x="成本情境", y=["誤報 FP", "漏報 FN"])
    st.caption("圖表只呈現三個已由 Validation 決定的成本情境，不把 Test Set 拿來重新挑門檻。")

    st.markdown("### 👥 有限人工審核能力：Top-K")
    t100 = metrics["top_k"]["100"]
    t300 = metrics["top_k"]["300"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision @ Top-100", f"{t100['precision_at_k']*100:.2f}%")
    c2.metric("Recall @ Top-100", f"{t100['recall_at_k']*100:.2f}%")
    c3.metric("Precision @ Top-300", f"{t300['precision_at_k']*100:.2f}%")
    c4.metric("Recall @ Top-300", f"{t300['recall_at_k']*100:.2f}%")
    st.caption("Top-K 模擬風控人力有限時，只優先審核風險排名最高的 K 筆交易。")

# ---------- Page 5 ----------
elif page == "🎚️ 風控策略模擬器":
    st.subheader("動態風控策略模擬器")
    st.info("💡 **怎麼看？** 你可以自己拉門檻。門檻改變後，警報數、TP、FP、FN、誤報率與加權成本會一起更新。")

    sim_cost = st.selectbox("模擬成本權重 FN:FP", ["5:1", "10:1", "20:1"], index=["5:1","10:1","20:1"].index(cost_label))
    sim_fn_cost = int(sim_cost.split(":")[0])
    sim_rec = float(thresholds[sim_cost])
    sim_th = st.slider("XGBoost 高風險門檻", 0.01, 0.99, float(round(sim_rec, 3)), 0.001, key="sim_th")
    sp, stn, sfp, sfn, stp = cm_at_threshold(sim_th)
    salert = int(sp.sum())
    s_prec = stp / max(stp+sfp, 1)
    s_rec = stp / max(stp+sfn, 1)
    s_fp10k = sfp / max(normal_n, 1) * 10000
    s_cost = sfn * sim_fn_cost + sfp

    if sim_th < sim_rec:
        direction = "門檻低於 Validation 推薦值：系統更積極攔截，通常會增加警報與 FP。"
    elif sim_th > sim_rec:
        direction = "門檻高於 Validation 推薦值：系統更保守，通常會減少 FP，但可能增加 FN。"
    else:
        direction = "目前使用 Validation Set 在此成本情境下選出的推薦門檻。"
    st.warning(direction)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("警報筆數", f"{salert:,}", f"{salert/total_n*100:.3f}%")
    r2.metric("TP / FP", f"{stp} / {sfp}")
    r3.metric("FN", sfn)
    r4.metric("每萬筆誤報", f"{s_fp10k:.2f}")

    r5, r6, r7 = st.columns(3)
    r5.metric("Precision", f"{s_prec*100:.2f}%")
    r6.metric("Recall", f"{s_rec*100:.2f}%")
    r7.metric(f"加權成本（{sim_cost}）", s_cost)

    st.markdown("### 🏦 建議處置邏輯")
    decision_df = pd.DataFrame({
        "風險區間": ["< 50%", "50% 至人工審核門檻", "人工審核門檻至 Block 門檻", f"≥ {sim_th*100:.1f}%"],
        "建議處置": ["Pass", "OTP / 3DS", "Manual Review", "Block"],
        "目的": ["降低正常客戶摩擦", "增加身分驗證", "交由人員判斷", "優先阻擋最高風險交易"],
    })
    st.dataframe(decision_df, use_container_width=True, hide_index=True)
    st.caption("此頁是 Test Set 上的離線決策模擬。真正部署時，門檻仍應先由 Validation 或後續期間資料決定。")

# ---------- Page 6 ----------
elif page == "🔍 XAI 模型解釋":
    st.subheader("XAI：模型為什麼做出這個判斷？")
    st.warning(
        "V1–V28 是 PCA 去識別化特徵。SHAP 可以告訴我們模型依賴哪些數學維度，但不能把 V14、V4 等直接說成"
        "『交易地點』『年齡』『消費習慣』等真實世界變數。"
    )

    st.markdown("### 🌍 XGBoost 全域 SHAP 重要性")
    imp = pd.DataFrame(list(shap_importance.items()), columns=["特徵", "平均 |SHAP|"]).head(15)
    st.bar_chart(imp.set_index("特徵"))
    st.dataframe(imp, use_container_width=True, hide_index=True)

    st.markdown("### 🔀 XGBoost 機率 vs Isolation Forest 異常分數")
    st.info(
        "**XGBoost Fraud Probability**：監督式模型輸出的詐欺機率。  \n"
        "**Isolation Forest Anomaly Score**：非監督式模型衡量一筆交易有多異常，不是詐欺機率。"
    )
    compare = df_eval[["xgb_prob", "iso_percentile", "Class"]].copy()
    compare["XGBoost 風險分數"] = compare["xgb_prob"] * 100
    compare["Isolation 異常百分位"] = compare["iso_percentile"] * 100
    st.scatter_chart(compare, x="XGBoost 風險分數", y="Isolation 異常百分位", color="Class")

    st.markdown("### ⏱️ 真實模型單筆推論延遲")
    if st.button("執行 100 次 XGBoost 單筆推論"):
        one = df_eval[feature_cols].iloc[[0]]
        lat = []
        for _ in range(100):
            t0 = time.perf_counter()
            xs = scaler.transform(one)
            _ = xgb_model.predict_proba(xs)[:, 1]
            lat.append((time.perf_counter() - t0) * 1000)
        l1, l2, l3 = st.columns(3)
        l1.metric("p50", f"{np.percentile(lat,50):.2f} ms")
        l2.metric("p95", f"{np.percentile(lat,95):.2f} ms")
        l3.metric("平均", f"{np.mean(lat):.2f} ms")
        st.caption("只代表目前 Streamlit 執行環境中的 scaler + XGBoost 推論時間，不等於銀行端到端系統延遲。")

# ---------- Page 7 ----------
elif page == "🎯 系統主要功能與應用對象":
    st.subheader("🎯 系統主要功能與應用對象")
    st.caption("從機器學習模型的預測結果，進一步轉換成可理解、可比較、可操作的信用卡交易風控資訊。")

    st.markdown("### 01｜我們在做什麼？")
    st.info(
        "本系統以**信用卡交易詐欺風險管理**為核心，將機器學習模型的預測結果轉換成可以實際理解的風控資訊。"
        "系統不只判斷交易是否可能涉及詐欺，也進一步整合**風險評分、成本敏感門檻、動態決策、模型比較與可解釋性分析**，"
        "協助使用者從「模型預測」走到「風控決策」。"
    )

    st.markdown(
        """
        <div style="
            padding: 18px 20px;
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 14px;
            text-align: center;
            font-size: 1.05rem;
            line-height: 2;
            margin-bottom: 12px;
        ">
        <b>交易資料</b>　→　<b>模型分析</b>　→　<b>風險分數</b>　→　
        <b>成本與門檻</b>　→　<b>風控處置</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 02｜五大主要功能")
    function_df = pd.DataFrame({
        "主要功能": [
            "即時風險評分",
            "多模型比較",
            "成本敏感分析",
            "動態風控決策",
            "模型解釋 XAI",
        ],
        "系統在做什麼": [
            "使用 XGBoost 計算交易的詐欺風險，並轉換成可閱讀的風險資訊。",
            "在相同 Test Set 比較 Logistic Regression、Isolation Forest 與 XGBoost。",
            "比較 FN:FP = 5:1、10:1、20:1，觀察不同風險成本下的門檻與結果。",
            "調整 Threshold 後，即時重新計算警報、TP、FP、FN、Recall、Precision 與成本。",
            "使用 SHAP 說明 XGBoost 的重要特徵，並區分詐欺機率與異常分數。",
        ],
        "可以回答的問題": [
            "哪些交易需要優先注意？",
            "不同模型的辨識效果有何差異？",
            "漏報與誤報成本改變時，門檻如何調整？",
            "攔截更嚴格或更寬鬆時，會產生什麼影響？",
            "模型主要依據哪些匿名化特徵做判斷？",
        ],
    })
    st.dataframe(function_df, use_container_width=True, hide_index=True)

    st.markdown("### 03｜誰會使用這套系統？")
    audience1, audience2, audience3 = st.columns(3)

    with audience1:
        st.markdown(
            """
            #### 🏦 金融機構風控人員
            觀察高風險交易、警報率與異常交易。

            系統可以協助風控人員辨識需要優先處理的交易，並依風險程度採取不同處置。
            """
        )

    with audience2:
        st.markdown(
            """
            #### 👨‍💼 風控主管與決策者
            比較漏報與誤報的成本。

            系統可以協助主管評估不同風險胃納下的 Threshold，並觀察門檻改變後的營運影響。
            """
        )

    with audience3:
        st.markdown(
            """
            #### 📊 模型與資料分析人員
            比較不同模型的預測表現。

            系統提供 PR-AUC、ROC-AUC、Precision、Recall、F1 與 SHAP 等資訊，協助檢視模型效果與解釋性。
            """
        )

    st.markdown("### 04｜從模型結果到銀行風控處置")
    process_df = pd.DataFrame({
        "風險階段": ["低風險", "中度風險", "較高風險", "最高風險"],
        "系統處置": ["Pass", "OTP / 3DS", "Manual Review", "Block"],
        "決策目的": [
            "正常授權，降低對一般客戶的干擾。",
            "增加一次身分驗證，降低可疑交易風險。",
            "交由風控人員進一步檢查。",
            "優先阻擋高風險交易並進一步確認。",
        ],
    })
    st.dataframe(process_df, use_container_width=True, hide_index=True)
    st.caption("四級處置為本研究的風控決策原型，用於呈現模型分數如何轉換成不同管理行動，不代表特定銀行的實際作業規則。")

    st.markdown("### 05｜這個網站的核心價值")
    st.success(
        "### 從「模型預測」走向「風控決策」\n\n"
        "一般模型分析常著重於哪個模型的預測表現較好。"
        "本系統進一步考慮**漏報、誤報、風險門檻與有限審核資源**，"
        "並將模型結果轉換成 **Pass、OTP / 3DS、Manual Review、Block** 等決策資訊。\n\n"
        "因此，本網站的重點不只是「找出詐欺」，也在呈現**金融機構如何根據模型資訊做風險管理決策**。"
    )

    st.markdown("### 06｜網站各頁面可以看到什麼？")
    guide_df = pd.DataFrame({
        "頁面": [
            "🏠 風控總覽",
            "⚡ 近即時交易回放",
            "🤖 模型競技場",
            "💰 成本敏感實驗室",
            "🎚️ 風控策略模擬器",
            "🔍 XAI 模型解釋",
        ],
        "主要用途": [
            "快速掌握真實詐欺率、模型警報率與整體風控結果。",
            "查看單筆交易如何從模型分數轉換成風控處置。",
            "公平比較三種模型在共同 Test Set 上的表現。",
            "了解不同 FN / FP 成本設定如何影響最佳門檻。",
            "自行調整 Threshold，觀察警報、誤報、漏報與成本變化。",
            "理解 XGBoost 的重要特徵，並區分詐欺機率與異常分數。",
        ],
    })
    st.dataframe(guide_df, use_container_width=True, hide_index=True)

    st.warning(
        "**系統定位：**本網站使用歷史信用卡交易資料建立風控決策研究原型，"
        "並以近即時交易回放方式展示模型應用。系統並未直接串接銀行核心交易系統，"
        "網站中的風控處置與損失估計屬研究模擬。"
    )

st.markdown("---")
st.caption(
    "研究限制：本系統使用 Kaggle 歷史信用卡交易資料建立離線研究與近即時交易回放原型，"
    "不是直接串接真實銀行核心系統。網站載入已完成離線訓練的模型，Test Set 不進行 SMOTE。"
)
