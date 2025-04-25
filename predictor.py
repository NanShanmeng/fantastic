import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. 加载模型和数据
model = joblib.load('XGBoost.pkl')

# 2. 定义特征名称和初始值
feature_names = [
    "SLC6A13", "ANLN", "MARCO", "SYT13", "ARG2", "MEFV", "ZNF29P",
    "FLVCR2", "PTGFR", "CRISP2", "EME1", "IL22RA2", "SLC29A4",
    "CYBB", "LRRC25", "SCN8A", "LILRA6", "CTD_3080P12_3", "PECAM1"
]

initial_values = [
    162, 1088, 33379, 43, 114, 440, 0, 3509, 907, 120,
    50, 9, 433, 36739, 1724, 71, 1528, 104, 94840
]

# 3. 设置网页标题和说明
st.title("Non-small Cell Lung Cancer Risk Prediction Model")
st.markdown("Assessing the Risk of Non-Small Cell Lung Cancer Based on Diabetes-Related Gene Expression Levels.")

# 5. 创建输入表单
st.sidebar.header("Gene Expression Level Settings")
inputs = {}
for feature, value in zip(feature_names, initial_values):
    inputs[feature] = st.sidebar.slider(
        feature,
        min_value=0.0,
        max_value=100000.0,
        value=float(value)
    )

# 6. 显示输入数据表格
st.subheader("Input Gene Expression Data")
input_df = pd.DataFrame([inputs.values()], columns=feature_names)
st.table(input_df.style.format("{:.1f}").highlight_max(axis=0))

# 7. 预测功能
if st.button("Calculate Disease Risk"):
    # 预测概率
    predicted_proba = model.predict_proba(input_df)[0]
    tumor_risk = predicted_proba[1] * 100
    
    # 显示结果
    st.subheader("Risk Assessment Results")
    if tumor_risk >= 50:
        risk_level = "High Risk"
        color = "#FF5733"
        class_idx = 1
    else:
        risk_level = "Low Risk"
        color = "#33C1FF"
        class_idx = 0

    # 显示风险概率和等级
    st.markdown(f"Predicted Probability: <span style='color:{color}; font-weight:bold;'>{tumor_risk:.2f}%</span>", 
                unsafe_allow_html=True)
    st.markdown(f"Risk Level: <span style='color:{color}; font-weight:bold;'>{risk_level}</span>", 
                unsafe_allow_html=True)

    # 医学建议
    advice = """
    ### Medical Advice:
    """ + ("We're sorry to inform you that, according to the model's prediction, you have a high risk of having the disease. It's advisable to contact a healthcare professional for a thorough examination at the earliest. Please note that our results are for reference only and cannot replace a professional diagnosis from a hospital." if tumor_risk >= 50 else "We're glad to inform you that, according to the model's prediction, your disease risk is low. If you experience any discomfort, it's still advisable to consult a doctor. Please maintain a healthy lifestyle and have regular medical check-ups. Wishing you good health.")
    st.markdown(advice)

    # SHAP解释
    st.subheader("SHAP Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # 获取对应类别的SHAP值
    if isinstance(shap_values, list):
        selected_shap = shap_values[class_idx][0]
        base_value = explainer.expected_value[class_idx]
    else:
        selected_shap = shap_values[0]
        base_value = explainer.expected_value
    
    # 绘制SHAP力图
    plt.figure()
    shap.force_plot(
        base_value=base_value,
        shap_values=selected_shap,
        features=input_df.iloc[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())
    plt.clf()