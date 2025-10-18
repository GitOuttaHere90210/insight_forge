### REQUIRES USING VENV - Run from terminal before running app.py
# cd "X:\AI class\07 Capstone Project\insight_forge"
# .\.venv\Scripts\Activate.ps1

import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import streamlit as st
import pandas as pd
from bi.data_loader import load_data
from bi.metrics import sales_by_month, sales_by_region, satisfaction_by_region, sales_by_gender, sales_trend_over_time
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.evaluation import load_evaluator  # Added back

# Optional Ragas imports
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset as HFDataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

load_dotenv()

st.title("InsightForge – Simple BI")
df = load_data()

# Session state initialization
if 'filter_history' not in st.session_state:
    st.session_state.update({'filter_history': [], 'summary_history': [], 'selected_region': None, 'selected_product': None})

# Filters
selected_region = st.selectbox("Filter by Region", ["All"] + sorted(df['Region'].unique().tolist()))
selected_product = st.selectbox("Filter by Product", ["All"] + sorted(df['Product'].unique().tolist()))

# Apply filters conditionally
mask = pd.Series(True, index=df.index)  # Default to all rows
if selected_region != "All":
    mask &= (df['Region'] == selected_region)
if selected_product != "All":
    mask &= (df['Product'] == selected_product)
df = df[mask]

# Track filter history
current_filters = {'Region': selected_region, 'Product': selected_product}
if current_filters not in st.session_state.filter_history:
    st.session_state.filter_history.append(current_filters)
    if len(st.session_state.filter_history) > 5:
        st.session_state.filter_history.pop(0)

# Display and download
st.subheader("Raw Data")
st.dataframe(df.head())
st.download_button("Download Filtered Data", df.to_csv(index=False), "filtered_sales_data.csv", "text/csv")

# Cached metrics
@st.cache_data
def get_metrics(_df):
    return {
        'month': sales_by_month(_df),
        'region': sales_by_region(_df),
        'satisfaction': satisfaction_by_region(_df),
        'gender': sales_by_gender(_df),
        'trend': sales_trend_over_time(_df)
    }

metrics = get_metrics(df)

# Visualizations
st.subheader("Sales by Month")
if not metrics['month'].empty: st.bar_chart(metrics['month'])
else: st.info("No sales data available")

st.subheader("Sales by Region")
if not metrics['region'].empty: st.bar_chart(metrics['region'])
else: st.info("No regional sales data available")

st.subheader("Satisfaction by Region")
if not metrics['satisfaction'].empty: st.bar_chart(metrics['satisfaction'])
else: st.info("No satisfaction data available")

st.subheader("Sales by Gender")
if not metrics['gender'].empty: st.bar_chart(metrics['gender'])
else: st.info("No gender sales data available")

st.subheader("Sales Trends Over Time")
if metrics['trend'] is not None and not metrics['trend'].empty:
    st.line_chart(metrics['trend'])
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Total Sales", f"${metrics['trend'].sum():,.2f}")
    with col2: st.metric("Average Sales", f"${metrics['trend'].mean():,.2f}")
    with col3: st.metric("Peak Sales", f"${metrics['trend'].max():,.2f}")
else:
    st.info("No trend data available")

# RAG Setup
api_key = os.getenv("OPENAI_API_KEY")
retriever = None

if api_key:
    @st.cache_resource
    def create_vectorstore(_df):
        def df_to_documents(df):
            documents = []
            for region, month in df.groupby(['Region', 'Month']).groups.keys():
                # Use a single mask to avoid chained indexing
                mask = (df['Region'] == region) & (df['Month'] == month)
                subset = df[mask]
                if not subset.empty:
                    doc = f"Region: {region}, Month: {month}\nTotal Sales: ${subset['Sales'].sum():,.2f}\n"
                    doc += f"Average Customer_Satisfaction: {subset['Customer_Satisfaction'].mean():.2f}\n"
                    doc += f"Products: {', '.join(subset['Product'].unique())}\n"
                    if not subset['Customer_Gender'].isna().all():
                        doc += f"Gender Distribution: {subset['Customer_Gender'].value_counts().to_dict()}\n"
                    documents.append(doc)
            return documents

        docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).create_documents(df_to_documents(_df))
        return FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=api_key))

    try:
        vectorstore = create_vectorstore(df)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        st.warning(f"Could not create vector store: {str(e)}")

# ... (previous code remains unchanged)

# Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    """You are a precise business analyst. Generate EXACTLY 4 bullet points using ONLY the numbers below.
CRITICAL RULES: Use only exact numbers, copy formatting, avoid calculations/inferences, skip unavailable data.
CURRENT FILTERS: - Region: {region} - Product: {product}
SALES BY MONTH: {month_totals}
PEAK SALES MONTH: {peak_month}  # Added explicit peak month
SALES BY REGION: {region_totals}
SATISFACTION BY REGION: {satisfaction}
SALES BY GENDER: {gender_totals}
INSTRUCTIONS: 1. Top region by sales 2. Trend from month data (use the peak sales month and value) 3. Satisfaction insight 4. Gender distribution
Format: - [Insight] [Recommendation]
Generate 4 bullet points:"""
)

@st.cache_data
def assemble_inputs(_=None):
    query = f"Sales data for {st.session_state.selected_region or 'all regions'} and {st.session_state.selected_product or 'all products'}"
    context = retriever.invoke(query) if retriever else "(no context)"
    context = "\n".join([d.page_content for d in context]) if context != "(no context)" else context

    def format_data(data, prefix=""):
        return "\n".join(f"  - {k}: {v}" for k, v in data.items()) if data else f"  {prefix}No data"

    # Calculate peak month and sales
    if not metrics['month'].empty:
        peak_month = metrics['month'].idxmax()
        peak_sales = f"${metrics['month'].max():,.2f}"
    else:
        peak_month = "No data"
        peak_sales = "No data"

    return {
        "context": context,
        "region": st.session_state.selected_region or "All",
        "product": st.session_state.selected_product or "All",
        "month_totals": format_data({k: f"${v:,.2f}" for k, v in metrics['month'].to_dict().items()}),
        "peak_month": f"{peak_month} at {peak_sales}",  # Added peak month with value
        "region_totals": format_data({k: f"${v:,.2f}" for k, v in metrics['region'].to_dict().items()}),
        "satisfaction": format_data({k: f"{v:.2f}" for k, v in metrics['satisfaction'].to_dict().items()}),
        "gender_totals": format_data({k: f"${v:,.2f}" for k, v in metrics['gender'].to_dict().items()})
    }

def generate_ground_truth():
    summary = []
    if not metrics['region'].empty:
        top_region = metrics['region'].idxmax()
        summary.append(f"- {top_region} leads with ${metrics['region'].max():,.2f} [Focus marketing here]")
    if not metrics['month'].empty:
        peak_month = metrics['month'].idxmax()
        summary.append(f"- Peak sales in {peak_month} at ${metrics['month'].max():,.2f} [Plan seasonal campaigns]")
    if not metrics['satisfaction'].empty:
        top_sat = metrics['satisfaction'].idxmax()
        summary.append(f"- {top_sat} has {metrics['satisfaction'].max():.2f} satisfaction [Enhance this region]")
    if not metrics['gender'].empty:
        gender_dist = ", ".join(f"{k}: ${v:,.2f}" for k, v in metrics['gender'].to_dict().items())
        summary.append(f"- Gender sales: {gender_dist} [Target key demographics]")
    return "\n".join(summary) if summary else "No data available"

# LLM and Evaluation
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0.1) if api_key else None

st.subheader("AI Summary")
if llm:
    try:
        inputs = assemble_inputs()  # Call without invoke for initial setup
        summary = (RunnableLambda(assemble_inputs) | prompt_template | llm | StrOutputParser()).invoke({})
        st.text(summary)
        st.session_state.summary_history.append({"filters": {"Region": selected_region, "Product": selected_product}, "summary": summary})
        if len(st.session_state.summary_history) > 5: st.session_state.summary_history.pop(0)
        ground_truth = generate_ground_truth()
        with st.expander("Debug"):
            st.text(f"Ground Truth: {ground_truth}")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Set OPENAI_API_KEY to enable AI summaries.")

# ... (rest of the code remains unchanged)

st.subheader("LangChain Evaluation")
if api_key and 'summary' in locals() and ground_truth:
    try:
        eval_llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, temperature=0)
        eval_result = load_evaluator("labeled_criteria", criteria="correctness", llm=eval_llm).evaluate_strings(
            prediction=summary, reference=ground_truth, input=prompt_template.format(**inputs))
        st.write(f"- Score: {eval_result.get('score', 'N/A')} - Reasoning: {eval_result.get('reasoning', 'N/A')}")
    except Exception as e:
        st.error(f"Evaluation error: {e}")
else:
    st.info("Generate a summary to enable evaluation.")

# st.subheader("Ragas Evaluation")
# if not RAGAS_AVAILABLE:
#     st.warning("Install ragas: pip install ragas datasets")
# elif api_key and 'summary' in locals() and ground_truth and ground_truth != "No data available":
#     try:
#         # Get raw context documents as a list
#         query = f"Summarize for Region={selected_region}, Product={selected_product}"
#         contexts = retriever.invoke(query) if retriever else []
#         retrieved_contexts = [doc.page_content for doc in contexts] if contexts else []

#         eval_data = {
#             "question": [query],
#             "answer": [summary],
#             "contexts": retrieved_contexts,  # Use list of context strings
#             "ground_truth": [ground_truth]
#         }
#         result = evaluate(HFDataset.from_dict(eval_data), [faithfulness, answer_relevancy, context_precision, context_recall])
#         result_df = result.to_pandas()
#         col1, col2 = st.columns(2)
#         with col1: 
#             st.metric("Faithfulness", f"{result_df['faithfulness'].iloc[0]:.3f}")
#             st.metric("Relevancy", f"{result_df['answer_relevancy'].iloc[0]:.3f}")
#         with col2: 
#             st.metric("Precision", f"{result_df['context_precision'].iloc[0]:.3f}")
#             st.metric("Recall", f"{result_df['context_recall'].iloc[0]:.3f}")
#     except Exception as e:
#         st.error(f"Ragas error: {e}")
# else:
#     st.info("Generate a summary with RAG to see evaluation.")

st.subheader("Filter History")
for i, filters in enumerate(st.session_state.filter_history[:5]):
    st.write(f"Selection {i+1}: {filters}")

### REQUIRES USING VENV - Run from terminal before running app.py
# cd "X:\AI class\07 Capstone Project\insight_forge"
# .\.venv\Scripts\Activate.ps1
# streamlit run app.py