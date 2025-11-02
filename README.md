# 🚀 FedEx Smart Returns Intelligence (ReturnIQ)

An AI-powered **decision intelligence engine** that helps FedEx and merchants **predict the profitability and sustainability of international product returns** — optimizing logistics costs and reducing environmental impact.

---

## 💡 Problem Statement

International product returns are a major challenge in global e-commerce:
- Reverse logistics are **expensive**, often exceeding the product’s original price.  
- Returns cause significant **carbon emissions** due to redundant transport.  
- Merchants lack intelligent tools to **decide whether a return is worth it**.  

FedEx aims to build smarter global return systems that minimize **cost**, **CO₂ footprint**, and **operational inefficiency** while maintaining customer satisfaction.

---

## 🎯 Objective

**ReturnIQ** empowers merchants and FedEx logistics planners to make data-driven decisions on returns by predicting:

- ✅ **When to accept a return**
- 💸 **When to refund without return**
- ❌ **When to reject a return**

Each decision balances:
- **Financial viability**
- **Carbon impact**
- **Operational risk**

---

## ⚙️ Solution Overview

ReturnIQ is an end-to-end prototype built using:
- **FedEx shipping zone data** (from rate PDFs)
- **Simulated product-level datasets**
- **Machine Learning** (Random Forest)
- **Streamlit dashboard** for interactive insights

The model estimates shipping cost, customs tariffs, CO₂ emissions, and resale potential to output a **Return Feasibility Score** and **actionable recommendation**.

---

## 🧠 Methodology

| Stage | Description |
|--------|-------------|
| **1. Feature Engineering** | Compute derived metrics: return cost ratio, resale ratio, CO₂ emissions, and CVaR risk (financial tail loss). |
| **2. Model Training** | Train a Random Forest Classifier to classify decisions into Accept / Refund / Reject. |
| **3. Visualization Layer** | A Streamlit app provides a clean UX with a FedEx-branded theme, KPI metrics, circular confidence gauge, and route map. |

---

## 🔢 Model Inputs

| Parameter | Description |
|------------|-------------|
| Product Price | Value of the sold item (no upper limit). |
| Weight (kg) | Product weight influencing shipping cost. |
| Origin & Destination Country | Used to fetch FedEx zone-based costs. |
| Tariff Rate | Customs duty applicable to returned goods. |
| Condition Score | 0–1 measure of resale condition. |
| Fragile / Valuable Flags | Adds cost & risk multipliers. |

---

## 📊 Outputs & KPIs

| Metric | Description |
|----------|-------------|
| **Decision** | Accept / Refund / Reject |
| **Confidence Score** | Model’s certainty shown in circular visualization |
| **Estimated Return Cost** | Computed from FedEx rate + surcharges |
| **CO₂ Emission (kg)** | Environmental impact per shipment |
| **Expected Profit Margin** | After adjusting for tariffs and risk |

---

## 🌍 Impact

| Dimension | Benefit |
|------------|----------|
| **Cost** | Reduces unnecessary international returns, saving logistics costs. |
| **Sustainability** | Minimizes avoidable CO₂ emissions and fuel waste. |
| **Customer Experience** | Enables instant refund-without-return options for low-value goods. |
| **Operational Efficiency** | Streamlines global return policies for FedEx merchants. |

---

## 🧩 System Architecture

