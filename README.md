# Credit Card Fraud Risk Decision Prototype

本專題以 Kaggle Credit Card Fraud Detection 歷史資料建立信用卡交易詐欺風控原型。

## GitHub / Streamlit 版本

為避免 GitHub 網頁上傳單檔大小限制，本 repository **不放完整 284,807 筆訓練資料**。
`demo_data.csv` 是研究流程中依時間排序後的最後 15% Test Set，共 42,722 筆交易，僅供 Streamlit 獨立測試集回放。模型與研究指標則由完整 284,807 筆資料離線訓練後儲存在 `artifacts/`。

## 專案結構

```
app.py
train_pipeline.py
requirements.txt
demo_data.csv
artifacts/
  xgb_model.json
  lr_model.pkl
  iso_model.pkl
  scaler.pkl
  feature_cols.json
  thresholds.json
  metrics.json
  shap_importance.json
```

## 直接啟動網站

```bash
pip install -r requirements.txt
streamlit run app.py
```

不需要重新訓練模型。

## 如需重現完整訓練

自行取得完整 `creditcard.csv` 後放在專案根目錄，再執行：

```bash
python train_pipeline.py
```

程式也支援拆成 `creditcard_part1.csv` 與 `creditcard_part2.csv`。

## 研究流程

完整資料依 `Time` 排序後切分為 Train 70%、Validation 15%、Test 15%。SMOTE 僅使用於訓練資料。Validation Set 用於成本敏感門檻選擇，Test Set 保留作最終評估。

網站載入已訓練的 XGBoost 模型，以 `predict_proba` 對 Test Set 交易進行風險評分，並展示模型比較、成本門檻、Top-K 人力限制與 SHAP 全域特徵重要性。

> 本系統為學術研究與風控決策原型，不是直接串接銀行核心系統的正式線上生產系統。
