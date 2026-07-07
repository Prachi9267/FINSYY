import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dotenv
from typing import List, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

st.set_page_config(page_title="Finsyy Portal", layout="wide")


class CleanedTransaction(BaseModel):
    date: str = Field(description="ISO Date format YYYY-MM-DD")
    description: str = Field(
        description=(
            "The cleaned description or merchant name"
            "If the narration contains words like 'Auto pay' include that also in description along with other words."
        )
    )
    category: str = Field(
        description=(
            "The broad structural category classification. "
            "MUST be exactly one of the specific keyword rule strings provided in the system prompt "
            "(e.g., 'Food & Dining', 'Travel & Fuel', 'Online Shopping', 'Utilities', 'Investments', 'Groceries'). "
            "If it does NOT match any keyword rules, do NOT use 'Others' or copy the full description; "
            "instead extract just the short name of the person to use as the category."
        )
    )

    payment_mode: Literal["UPI", "NetBanking", "Card", "Check", "Cash"] = Field(
        description="The transaction medium used."
    )

    amount: float = Field(
        description="Negative for debits/expenses, positive for credits/income"
    )


class StatementBatchResponse(BaseModel):
    transactions: List[CleanedTransaction]
    total_credit: float = Field(
        description="Total of all the positive values in amount."
    )
    total_debit: float = Field(
        description="Total of all the negative values in amount , in the end give the final ans without minus sign."
    )
    highest_amount: float = Field(
        description="The positive version of highest  value where the amount is negative."
    )
    highest_cat: str = Field(
        description="The category associated with the highest debit amount."
    )


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def process_statement_file(uploaded_file):
    """
    Accepts a streamlit uploadedfile object directly from the web interface.
    """

    raw_table_string = "..."

    filename = uploaded_file.name

    if filename.endswith("xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    elif filename.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please use CSV or XLSX.")

    df = df.dropna(how="all")
    raw_table_string = df.to_string(index=False)

    system_instructions = (
        "You are a financial analysis assistant. Your job is to extract bank transactions "
        "and enforce strict categorisation rules. For the 'category' field, you MUST prioritize "
        "the following keyword mapping rules, paying close attention to specific eceptions: \n\n"
        "- 'Food & Dining': If the narration contains ZOMATO, SWIGGY, FOOD, RESTAURANT, SNACKS, CAFE, DHABA, KITCHEN, or HALDI RAM.\n"
        "- 'Travel & Fuel': If the narration contains DMRC, UBER, OLA, CAB, PETROL, FUEL, TRAVEL, or RAPIDO.\n"
        "- 'Online Shopping': If the narration contains AMAZON, FLIPKART, MYNTRA, AJIO, SHOP, NYKAA, H AND M, HANDM, TIRA BEAUTY, TIRABEAUTY, ZUDIO or TIRA.\n"
        "  * EXCEPTION 1: If the narration contains 'AWS', 'AMAZON WEB SERVICES', or 'AMAZON CLOUD', it is a cloud infrastructure expense. Classify it under 'Utilities' and clean the description to 'AWS'.\n"
        "  * EXCEPTION 2: If the narration contains 'AMAZON PAY GROCERIES' or 'AMAZON FRESH', classify it under 'Groceries' instead of Online Shopping.\n"
        "- 'Utilities': If the narration contains ELECTRICITY, WATER, BILL, RECHARGE, AIRTEL, RENT or INTERNET, or cloud infrastructure exceptions like AWS.\n"
        "- 'Investments': If the narration contains INSURANCE, SIP, MUTUAL FUND, or GROWW.\n"
        "- 'Groceries': If the narration contains BLINKIT, BIGBASKET, GROCERY, MART, ZEPTO, INSTAMART, DMART or ALL MART, or 'AMAZON OAY GROCERIES'.\n\n"
        "If a narration does not contain any of the keywords listed above, DO NOT classify it as 'Others'. "
        "CRITICAL NAME RULE: For the 'description' field, if the transaction involves an individual's name "
        "(like Rahul Sharma), DO NOT alter, change, or guess the name. Keep the person's name EXACTLY "
        "as it appears in the raw narration, simply stripping away things like 'UPI/' or transaction IDs around it. "
        "Do not substitute one human name for another."
        "Do NOT use 'UPI' as a category name under any circumstance. UPI is a payment mode, not a spending category. "
        "The category field must hold ONLY the short clean grouping name. "
        "Do not copy the full narration string into the category field.\n\n"
    )
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_instructions},
            {
                "role": "user",
                "content": f"Here is the raw sheet data:\n\n{raw_table_string}",
            },
        ],
        response_format=StatementBatchResponse,
    )

    parsed_data = response.choices[0].message.parsed
    tx_list = [tx.model_dump() for tx in parsed_data.transactions]
    tx_df = pd.DataFrame(tx_list)

    total_credit = tx_df[tx_df["amount"] > 0]["amount"].sum()
    total_debit = abs(tx_df[tx_df["amount"] < 0]["amount"].sum())

    debit_df = tx_df[tx_df["amount"] < 0].copy()
    if not debit_df.empty:
        debit_df["debit_amount"] = debit_df["amount"].abs()

        # Single highest transaction calculation
        idx_max = debit_df["debit_amount"].idxmax()
        highest_amount = debit_df.loc[idx_max, "debit_amount"]
        highest_cat = debit_df.loc[idx_max, "category"]

        category_totals = (
            debit_df.groupby("category")["debit_amount"].sum().reset_index()
        )
        category_totals = category_totals.sort_values(
            by="debit_amount", ascending=False
        )
    else:
        highest_amount = 0.0
        highest_cat = "N/A"
        category_totals = pd.DataFrame(columns=["category", "debit_amount"])

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.set_theme(style="whitegrid")

    sns.barplot(
        x="category",
        y="debit_amount",
        data=category_totals,
        palette="Reds_r",
        hue="category",
        legend=False,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9, rotation=0)
    plt.title(
        "Total Debit Spending by Category", fontsize=14, fontweight="bold", pad=15
    )
    plt.xlabel("Categories", fontsize=12)
    plt.ylabel("Total Amount Spent (Debited)", fontsize=12)
    plt.xticks(rotation=90, ha="center", va="top", fontsize=10)
    plt.tight_layout()

    st.pyplot(fig)

    return {
        "total_credit": total_credit,
        "total_debit": total_debit,
        "highest_amount": highest_amount,
        "highest_cat": highest_cat,
    }


st.title("FINSYY")
st.write("Analyze and improve your relationship with money.")

uploaded_file = st.file_uploader(
    "Drop your bank statement here(XLSX, XLS, OR CSV)", type=["xls", "xlsx", "csv"]
)

if uploaded_file is not None:
    st.success(f"Loaded: {uploaded_file.name}")

    # Process Button execution trigger
    if st.button("Categorize Statement Data", type="primary"):
        with st.spinner(
            "Analyzing statement structures and executing processing engine..."
        ):
            try:
                # Execute engine
                result = process_statement_file(uploaded_file)

                # Display Summary Metric Scorecard Layout
                st.markdown("### Executive Financial Summary")
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric(
                        label="Total Inflow (Credit)",
                        value=f"₹{result['total_credit']:,.2f}",
                    )
                with m_col2:
                    st.metric(
                        label="Total Outflow (Debit)",
                        value=f"₹{result['total_debit']:,.2f}",
                    )
                with m_col3:
                    st.metric(
                        label="Highest Single Outflow",
                        value=f"₹{result['highest_amount']:,.2f}",
                        delta=result["highest_cat"],
                        delta_color="inverse",
                    )

                # Render Cleaned Interactive Data Table
                st.markdown("### Processed Transaction Directory")
                st.dataframe(result["df"], use_container_width=True)

            except Exception as e:
                st.error(f"Execution crashed: {e}")
else:
    st.info("Awaiting structural document upload to run categorization pipelines.")
