#PART A
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import google.generativeai as genai
import os
import io
from fpdf import FPDF
import base64
from time_category_insights import time_category_insights_ui









## NEW
try:
    import tabula
    if not hasattr(tabula, 'read_pdf'):
        raise ImportError("tabula-py is not installed or the wrong 'tabula' package is installed. Please install tabula-py via 'pip install tabula-py' and ensure Java is available.")
    TABULA_AVAILABLE = True
except ImportError as e:
    TABULA_AVAILABLE = False
    TABULA_IMPORT_ERROR = str(e)









# --- Ensure prompts dict is initialized and contains all required keys at the top ---
if 'prompts' not in globals():
    prompts = {}
if "budget_creation" not in prompts:
    prompts["budget_creation"] = '''
You are a financial budget assistant. Create a personalized budget based on the user's spending history and financial goals.

Instructions:
- Analyze the historical spending patterns across categories
- Consider seasonal variations and spending clusters
- Create a realistic budget that helps achieve the user's financial goals
- Provide explanations for your recommendations
- Suggest specific areas where spending can be reduced based on patterns
- Include both fixed expenses and discretionary spending
- Format the budget in a clear, structured way
- Return the budget as a JSON array of objects, each with: category, allocation, type (fixed/discretionary), and a short note. Example:
[
  {{"category": "Rent", "allocation": 20000, "type": "fixed", "note": "Based on your past rent payments."}},
  {{"category": "Groceries", "allocation": 8000, "type": "discretionary", "note": "Average monthly grocery spend."}}
]
- After the JSON, provide a brief summary and recommendations.

User's spending history:
{category_spending_data}

Spending patterns by time:
{temporal_patterns}

Cluster information:
{cluster_summary}

Recent transactions:
{recent_activity}

User's financial goals: {goals}
User's income (if provided): {income}
Budget timeframe: {timeframe}

Create a detailed, personalized budget with specific allocation amounts for each category.
'''






#PART B
# Streamlit UI
st.title("M-PESA Transactions Dashboard")
# File uploader for XLSX (move before load_data)
st.sidebar.header("Upload M-PESA Statement (XLSX or PDF)")
file = st.sidebar.file_uploader("Upload XLSX or PDF Statement", type=["xlsx", "pdf"])
pdf_password = None
if file is not None and file.name.lower().endswith(".pdf"):
    pdf_password = st.sidebar.text_input("PDF Password (if required)", type="password")

if file is None:
    st.warning("You must upload an M-PESA XLSX or PDF statement to use this dashboard.")
    st.stop()










#PART C
def categorize_details(details):
    details = str(details).lower()
    if any(word in details for word in ["airtime", "tingg", "safaricom", "airtel", "bundles", "gessy"]):
        return "Airtime"
    elif any(word in details for word in ["kplc"]):
        return "Power"
    elif any(word in details for word in ["7629905"]):
        return "Rent Payment"
    elif any(word in details for word in ["cleanshelf", "equity", "kcb","naivas", "tuskys", "quick mart", "carrefour", "4093275","supermarket", "shopping", "small business", "mall","jumia", "kilimall", "amazon", "shop", "market", "merchant", "direct pay"]):
        return "Shopping"
    elif any(word in details for word in ["baraka", "java", "hotel", "restaurant", "cafe"]):
        return "Restaurant"
    elif any(word in details for word in ["sacco", "uber"]):
        return "Transport"
    elif any(word in details for word in ["alpha", "water"]):
        return "Water"
    elif any(word in details for word in ["butchery", "meat", "butcher"]):
        return "Butchery"
    elif any(word in details for word in ["customer transfer"]):
        return "People Transfer"
    elif any(word in details for word in ["withdraw"]):
        return "Withdrawals"
    elif any(word in details for word in ["charge"]):
        return "Transaction Charge"
    elif any(word in details for word in ["pay bill"]):
        return "Pay Bill"
    else:
        return "Other"





#PART D
# Load data
@st.cache_data
def load_data(file=None, pdf_password=None):
    if file is not None:
        if file.name.lower().endswith(".xlsx"):
            # XLSX logic
            required_columns = [
                'Receipt No.',
                'Completion Time',
                'Details',
                'Transaction Status',
                'Paid In',
                'Withdrawn',
                'Balance'
            ]
            all_sheets = pd.read_excel(file, sheet_name=None)
            matching_tables = []
            for df in all_sheets.values():
                if all(col in df.columns for col in required_columns):
                    matching_tables.append(df[required_columns])
            if matching_tables:
                df = pd.concat(matching_tables, ignore_index=True)
            else:
                st.error("No tables with the required columns were found in the uploaded XLSX file.")
                return None
        elif file.name.lower().endswith(".pdf"):
            # PDF logic: Try tabula-py first, fallback to PyPDF2 text extraction
            if TABULA_AVAILABLE:
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                        tmp_pdf.write(file.read())
                        tmp_pdf.flush()
                        pdf_path = tmp_pdf.name
                    dfs = tabula.read_pdf(pdf_path, pages='all', password=pdf_password if pdf_password else None, multiple_tables=True)
                    required_columns = [
                        'Receipt No.',
                        'Completion Time',
                        'Details',
                        'Transaction Status',
                        'Paid In',
                        'Withdrawn',
                        'Balance'
                    ]
                    matching_tables = []
                    for df in dfs:
                        if all(col in df.columns for col in required_columns):
                            matching_tables.append(df[required_columns])
                    if matching_tables:
                        df = pd.concat(matching_tables, ignore_index=True)
                    else:
                        st.error("No tables with the required columns were found in the uploaded PDF file.")
                        return None
                except Exception as e:
                    if 'Incorrect password' in str(e) or 'Password required' in str(e):
                        st.error("Incorrect PDF password. Please check and try again.")
                    elif 'read_pdf' in str(e) or 'attribute' in str(e):
                        st.error("tabula-py is not installed or the wrong 'tabula' package is installed. Please install tabula-py via 'pip install tabula-py' and ensure Java is available.")
                    else:
                        st.error(f"Failed to extract tables from PDF: {e}")
                    return None
            elif 'TABULA_IMPORT_ERROR' in globals():
                st.error(f"tabula-py import error: {TABULA_IMPORT_ERROR}")
                return None
            else:
                # Fallback: Try to extract text with PyPDF2 and warn user
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    if pdf_reader.is_encrypted:
                        if pdf_password:
                            try:
                                pdf_reader.decrypt(pdf_password)
                            except Exception:
                                st.error("Incorrect PDF password. Please check and try again.")
                                return None
                        else:
                            st.error("This PDF is password-protected. Please enter the password.")
                            return None
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    st.warning("Tabular extraction from PDF is limited without tabula-py. Only raw text extracted. Please install tabula-py and Java for best results.")
                    return None
                except Exception as e:
                    if 'incorrect password' in str(e).lower():
                        st.error("Incorrect PDF password. Please check and try again.")
                    else:
                        st.error(f"Failed to read PDF: {e}")
                    return None
        else:
            st.error("Unsupported file type. Please upload an XLSX or PDF statement.")
            return None
    else:
        # Fallback to data.csv
        df = pd.read_csv('data.csv')
    df['Details'] = df['Details'].fillna('Other')
    df['Paid In'] = pd.to_numeric(df['Paid In'], errors='coerce').fillna(0)
    df['Withdrawn'] = pd.to_numeric(df['Withdrawn'], errors='coerce').fillna(0)
    df['Type'] = df['Withdrawn'].apply(lambda x: 1 if x == 0.0 else 0)
    df = df[df['Type'] == 0]
    df['Category'] = df['Details'].apply(categorize_details)
    df['Hour'] = pd.to_datetime(df['Completion Time']).dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['Completion Time']).dt.day_name()
    df['Month'] = pd.to_datetime(df['Completion Time']).dt.month
    df['Amount'] = df['Paid In'] - df['Withdrawn']
    return df






df = load_data(xlsx_file)
if df is None:
    st.stop()







#PART E
st.header("Raw Data")
st.dataframe(df.head(5))


df_clean = df.copy()
df_clean['Details'] = df_clean['Details'].fillna('Other')
df_clean['Paid In'] = df_clean['Paid In'].fillna(0)
df_clean['Withdrawn'] = df_clean['Withdrawn'].fillna(0)
df_clean['Type'] = df_clean['Withdrawn'].apply(lambda x: 1 if x == 0.0 else 0)
df_clean = df_clean[df_clean['Type'] == 0]
df_clean['Category'] = df_clean['Details'].apply(categorize_details)
df_clean['Hour'] = pd.to_datetime(df_clean['Completion Time']).dt.hour
df_clean['DayOfWeek'] = pd.to_datetime(df_clean['Completion Time']).dt.day_name()
df_clean['Month'] = pd.to_datetime(df_clean['Completion Time']).dt.month
df_clean['Amount'] = df_clean['Paid In'] - df_clean['Withdrawn']
category_dummies = pd.get_dummies(df_clean['Category'], prefix='Category')
category_dummies = category_dummies.astype(int).add_prefix('Converted_')
df_clean = pd.concat([df_clean,  category_dummies], axis=1)
dayofweek_dummies = pd.get_dummies(df_clean['DayOfWeek'], prefix='DayOfWeek')
dayofweek_dummies = dayofweek_dummies.astype(int).add_prefix('Converted_')
df_clean = pd.concat([df_clean,  dayofweek_dummies], axis=1)
hour_dummies = pd.get_dummies(df_clean['Hour'], prefix='Hour')
hour_dummies = hour_dummies.astype(int).add_prefix('Converted_')
df_clean = pd.concat([df_clean,  hour_dummies], axis=1)
month_dummies = pd.get_dummies(df_clean['Month'], prefix='Month')
month_dummies = month_dummies.astype(int).add_prefix('Converted_')
df_clean = pd.concat([df_clean,  month_dummies], axis=1)
encoded_columns = [col for col in df_clean.columns if col.startswith('Converted_')]
clustering_data = pd.concat([df_clean[encoded_columns], df_clean[['Amount']]], axis=1)


scaler = StandardScaler()
clustering_data['amount'] = scaler.fit_transform(clustering_data[['Amount']])


silhouette_scores = []
cluster_range = range(2, 11)  # Test cluster sizes from 2 to 10
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(clustering_data)
    silhouette_avg = silhouette_score(clustering_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)
optimal_clusters = cluster_range[silhouette_scores.index(max(silhouette_scores))]


kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_clean['purpose_cluster'] = kmeans.fit_predict(clustering_data)



from sklearn.metrics import silhouette_score
X_scaled = clustering_data.values  # Use the scaled data for silhouette score calculation
labels = kmeans.predict(X_scaled)
silhouette_avg = silhouette_score(X_scaled, labels)
st.subheader("Silhouette Score")
st.write(f'Silhouette Score: {silhouette_avg}')


cluster_summary = df_clean.groupby('purpose_cluster').agg(
    avg_amount=('Amount', 'mean'),
    median_amount=('Amount', 'median'),
    total_amount=('Amount', 'sum'),  
    count=('purpose_cluster', 'size')
)


def get_top_n_columns(cluster_data, prefix, n=2):
    cols = [col for col in cluster_data.columns if col.startswith(prefix)]
    counts = cluster_data[cols].sum().sort_values(ascending=False)
    # Remove prefix and get unique top n
    unique_top = []
    for col in counts.index:
        name = col.replace('Converted_', '')
        if name not in unique_top:
            unique_top.append(name)
        if len(unique_top) == n:
            break
    return ", ".join(unique_top)

top_categories = []
top_months = []
top_hours = []
top_days = []
for cluster in df_clean['purpose_cluster'].unique():
    cluster_data = df_clean[df_clean['purpose_cluster'] == cluster]
    top_categories.append(get_top_n_columns(cluster_data, 'Converted_Category'))
    top_months.append(get_top_n_columns(cluster_data, 'Converted_Month'))
    top_hours.append(get_top_n_columns(cluster_data, 'Converted_Hour'))
    top_days.append(get_top_n_columns(cluster_data, 'Converted_DayOfWeek'))

cluster_summary['top_categories'] = top_categories
cluster_summary['top_months'] = top_months
cluster_summary['top_hours'] = top_hours
cluster_summary['top_days'] = top_days

# Remove 'Category_Other' from the top_categories column in the summary (so it doesn't show in the output)
cluster_summary['top_categories'] = cluster_summary['top_categories'].apply(
    lambda x: ", ".join([cat for cat in x.split(", ") if cat != "Category_Other"])
)


# Display the cluster summary on the dashboard
st.subheader("Cluster Summary Table")
st.dataframe(cluster_summary)
























#PART F
# Bar chart of Hour of the Day vs Total Amount Spent
st.subheader("Total Amount Spent by Hour of the Day")
amount_by_hour = df_clean.groupby('Hour')['Amount'].sum().reindex(range(24))
fig3, ax3 = plt.subplots()
sns.barplot(x=amount_by_hour.index, y=amount_by_hour.values, palette='viridis')
ax3.set_xlabel('Hour of the Day')
ax3.set_ylabel('Total Amount Spent')
ax3.set_title('Total Amount Spent by Hour of the Day')
plt.xticks(range(24))
plt.tight_layout()
st.pyplot(fig3)



# Bar chart of Day of the Week vs Total Amount Spent
amount_by_day = df_clean.groupby('DayOfWeek')['Amount'].sum().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
fig2, ax2 = plt.subplots()
sns.barplot(x=amount_by_day.index, y=amount_by_day.values, palette='viridis')
ax2.set_xlabel('Day of the Week')
ax2.set_ylabel('Total Amount Spent')
ax2.set_title('Total Amount Spent by Day of the Week')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)




#C. Bar chart of Category vs Total Amount Spent
amount_by_category = df_clean.groupby('Category')['Amount'].sum().sort_values(ascending=False)
fig2, ax2 = plt.subplots()
sns.barplot(x=amount_by_category.index, y=amount_by_category.values, palette='viridis', ax=ax2)
ax2.set_title('Total Amount Spent by Category')
ax2.set_xlabel('Category')
ax2.set_ylabel('Total Amount Spent')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig2)
















time_category_insights_ui(df_clean)



# --- PART G: Conversational Interface (Gemini, Interactive, with Graph Data) ---
# You may want to set your API key securely in production
api_key =  os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# --- Enrich Gemini's data bank: Add sample rows from each cluster ---
cluster_samples = []
for cluster_id in cluster_summary.index:
    cluster_df = df_clean[df_clean['purpose_cluster'] == cluster_id]
    n = len(cluster_df)
    # Get first 5 rows
    first_rows = cluster_df.head(5)
    # Get middle 5 rows if enough data
    if n > 10:
        mid_start = max((n // 2) - 2, 0)
        mid_end = min(mid_start + 5, n)
        middle_rows = cluster_df.iloc[mid_start:mid_end]
    else:
        middle_rows = pd.DataFrame()
    # Get last 5 rows
    last_rows = cluster_df.tail(5)
    # Format for Gemini
    sample_text = f"\n---\nCLUSTER {cluster_id} SAMPLES\n"
    sample_text += f"First 5 rows:\n{first_rows.to_string(index=False)}\n"
    if not middle_rows.empty:
        sample_text += f"Middle 5 rows:\n{middle_rows.to_string(index=False)}\n"
    sample_text += f"Last 5 rows:\n{last_rows.to_string(index=False)}\n"
    cluster_samples.append(sample_text)
clusters_sample_bank = "\n".join(cluster_samples)

# --- Add Recent Activity (last 10 transactions) ---
recent_activity = df_clean.sort_values('Completion Time').tail(10)
recent_activity_text = recent_activity.to_string(index=False)

# Update Gemini prompt to include the sample bank and recent activity
prompts = {
    "cluster_qa": '''
You are a financial data assistant. Use the following cluster summary table, graph data, cluster sample rows, and recent activity to answer the user's question.

Instructions:
- First, summarize the key differences between the clusters in plain, simple language.
- Then, provide actionable and practical advice for the user's future spending, based on the patterns you see in the data.
- Focus on insights and recommendations, not just restating the tables.
- If the user asks for advice, give clear, specific suggestions for improving spending habits.
- If you notice any unusual or concerning patterns, mention them.
- Use the sample rows for each cluster and the recent activity to provide more context and examples if needed.
- Be concise, friendly, and helpful.

Cluster summary table:
{cluster_summary}

Total Amount Spent by Hour of the Day:
{graph_hour_table}

Total Amount Spent by Day of the Week:
{graph_day_table}

Total Amount Spent by Category:
{graph_category_table}

Sample rows from each cluster:
{clusters_sample_bank}

Recent Activity (last 10 transactions):
{recent_activity}

User question: {question}
'''
    # Add more prompt templates here as needed
}

st.markdown("""
<hr style='border: 2px solid #4F8BF9; margin-top: 40px; margin-bottom: 20px;'>
<h2 style='color: #4F8BF9;'>💬 Ask Gemini About Your Spending Patterns</h2>
<p style='font-size: 1.1em;'>Type a question about your clusters or spending graphs below. Gemini will answer using your cluster summary and the visualized data.</p>
""", unsafe_allow_html=True)

# Prepare graph data as text tables
amount_by_hour = df_clean.groupby('Hour')['Amount'].sum().reindex(range(24))
graph_hour_table = amount_by_hour.to_string()

amount_by_day = df_clean.groupby('DayOfWeek')['Amount'].sum().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
graph_day_table = amount_by_day.to_string()

amount_by_category = df_clean.groupby('Category')['Amount'].sum().sort_values(ascending=False)
graph_category_table = amount_by_category.to_string()

def ask_gemini_about_clusters(question, cluster_summary, graph_hour_table, graph_day_table, graph_category_table, clusters_sample_bank, recent_activity):
    prompt = prompts["cluster_qa"].format(
        cluster_summary=cluster_summary.to_string(),
        graph_hour_table=graph_hour_table,
        graph_day_table=graph_day_table,
        graph_category_table=graph_category_table,
        clusters_sample_bank=clusters_sample_bank,
        recent_activity=recent_activity,
        question=question
    )
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] Gemini failed: {e}"

with st.expander("💬 Gemini Conversational Assistant", expanded=True):
    st.markdown("""
    <style>
    .stTextInput>div>div>input {
        border: 2px solid #4F8BF9;
        border-radius: 8px;
        font-size: 1.1em;
        padding: 8px;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 8px;
        font-size: 1.1em;
        padding: 6px 18px;
        margin-top: 8px;
    }
    .stMarkdown h4 {
        color: #4F8BF9;
    }
    .gemini-answer-box {
        background: #f6f8fa;
        padding: 16px;
        border-radius: 8px;
        font-size: 1.1em;
        color: #222;
        margin-top: 10px;
        border: 1px solid #e1e4e8;
        word-break: break-word;
        white-space: pre-wrap;
    }
    </style>
    """, unsafe_allow_html=True)
    user_question = st.text_input("Ask a question about your clusters or graphs (e.g., 'Which cluster spends the most on shopping?'):", "What is my biggest spending category?")
    if st.button("Ask Gemini"):
        with st.spinner("Gemini is thinking..."):
            answer = ask_gemini_about_clusters(
                user_question, cluster_summary, graph_hour_table, graph_day_table, graph_category_table, clusters_sample_bank, recent_activity_text
            )
        st.markdown(f"<h4>Gemini's Answer:</h4>", unsafe_allow_html=True)
        st.markdown(f"<div class='gemini-answer-box'>{answer}</div>", unsafe_allow_html=True)

# ===================== GEMINI BUDGET CREATION FEATURE (MODULAR BLOCK) =====================

# Ensure prompts dict has the budget creation prompt
if 'prompts' not in globals():
    prompts = {}
if "budget_creation" not in prompts:
    prompts["budget_creation"] = '''
You are a financial budget assistant. Create a personalized budget based on the user's spending history and financial goals.

Instructions:
- Analyze the historical spending patterns across categories
- Consider seasonal variations and spending clusters
- Create a realistic budget that helps achieve the user's financial goals
- Provide explanations for your recommendations
- Suggest specific areas where spending can be reduced based on patterns
- Include both fixed expenses and discretionary spending
- Format the budget in a clear, structured way
- Return the budget as a JSON array of objects, each with: category, allocation, type (fixed/discretionary), and a short note. Example:
[
  {{"category": "Rent", "allocation": 20000, "type": "fixed", "note": "Based on your past rent payments."}},
  {{"category": "Groceries", "allocation": 8000, "type": "discretionary", "note": "Average monthly grocery spend."}}
]
- After the JSON, provide a brief summary and recommendations.

User's spending history:
{category_spending_data}

Spending patterns by time:
{temporal_patterns}

Cluster information:
{cluster_summary}

Recent transactions:
{recent_activity}

User's financial goals: {goals}
User's income (if provided): {income}
Budget timeframe: {timeframe}

Create a detailed, personalized budget with specific allocation amounts for each category.
'''

def prepare_category_spending_data(df):
    monthly = df.copy()
    monthly['YearMonth'] = pd.to_datetime(monthly['Completion Time']).dt.to_period('M')
    cat_month = monthly.groupby(['YearMonth', 'Category'])['Amount'].sum().unstack(fill_value=0)
    avg_monthly = cat_month.mean().round(2)
    return avg_monthly.to_string()

def prepare_temporal_patterns(df):
    by_day = df.groupby('DayOfWeek')['Amount'].sum().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    by_hour = df.groupby('Hour')['Amount'].sum().reindex(range(24))
    return f"By Day of Week:\n{by_day.to_string()}\n\nBy Hour:\n{by_hour.to_string()}"

def prepare_recent_activity(df):
    recent = df.sort_values('Completion Time').tail(10)
    return recent.to_string(index=False)

def budget_table_to_pdf(df_budget):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Gemini-Powered Budget Table", ln=True, align='C')
    pdf.ln(10)
    col_width = pdf.w / (len(df_budget.columns) + 1)
    for col in df_budget.columns:
        pdf.cell(col_width, 10, col, border=1)
    pdf.ln()
    for _, row in df_budget.iterrows():
        for val in row:
            pdf.cell(col_width, 10, str(val), border=1)
        pdf.ln()
    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    return pdf_output

# --- Improved Gemini Budget UI ---
def parse_budget_json_and_summary(response_text):
    import json, re
    match = re.search(r'(\[.*?\])', response_text, re.DOTALL)
    total_budget = None
    summary = None
    structured_summary = None
    if match:
        try:
            budget_table = json.loads(match.group(1))
            def normalize_key(k):
                return str(k).strip().strip('"\'').lower()
            normalized = []
            for item in budget_table:
                norm_item = {normalize_key(k): v for k, v in item.items()}
                normalized.append(norm_item)
            # Extract summary and total after JSON
            after_json = response_text[match.end():].strip()
            # Look for TOTAL_BUDGET: <number>
            total_match = re.search(r'TOTAL_BUDGET\s*[:=]\s*([\d,.]+)', after_json)
            if total_match:
                try:
                    total_budget = float(total_match.group(1).replace(',', ''))
                except Exception:
                    total_budget = None
            # Remove TOTAL_BUDGET line from summary
            summary_text = re.sub(r'TOTAL_BUDGET\s*[:=].*', '', after_json).strip()
            # Parse structured summary
            summary_match = re.search(r'SUMMARY:(.*)', summary_text, re.DOTALL)
            if summary_match:
                summary_block = summary_match.group(1)
                # Extract each bullet point
                key_savings = re.search(r'Key Savings Opportunities\s*[:\-]\s*(.*)', summary_block)
                biggest_spending = re.search(r'Biggest Spending Categories\s*[:\-]\s*(.*)', summary_block)
                warnings = re.search(r'Warnings/Advice\s*[:\-]\s*(.*)', summary_block)
                structured_summary = {
                    'Key Savings Opportunities': key_savings.group(1).strip() if key_savings else '',
                    'Biggest Spending Categories': biggest_spending.group(1).strip() if biggest_spending else '',
                    'Warnings/Advice': warnings.group(1).strip() if warnings else ''
                }
            summary = summary_text
            return normalized, total_budget, summary, structured_summary
        except Exception:
            return None, None, None, None
    return None, None, None, None

with st.expander("📝 Budget Creation Assistant", expanded=True):
    goals = st.text_area("Describe your financial goals (e.g., save more, reduce eating out, pay off debt):", "Save more and reduce unnecessary spending.")
    income = st.text_input("Enter your monthly income (optional):", "")
    target_amount = st.text_input("Enter your target budget amount (optional):", "")
    timeframe = st.selectbox("Select your budget timeframe:", ["Monthly", "Weekly"])
    show_pie = st.checkbox("Show Pie Chart of Budget Allocations", value=False)
    if st.button("Generate Budget with Gemini"):
        with st.spinner("Gemini is creating your budget..."):
            # Add target_amount and total budget instructions to prompt
            prompt = prompts["budget_creation"]
            prompt += "\nIf a target budget amount is provided, create a budget that fits within that amount. If it is not possible, explain why and suggest the closest achievable budget.\n"
            prompt += "Return the total budget allocation as: TOTAL_BUDGET: <number> on a new line after the JSON.\n"
            prompt += "After the JSON, provide a structured summary in this format:\nSUMMARY:\n- Key Savings Opportunities: <short point>\n- Biggest Spending Categories: <short point>\n- Warnings/Advice: <short point>\n"
            prompt += "Target budget amount: {target_amount}\n"
            category_data = prepare_category_spending_data(df_clean)
            temporal_data = prepare_temporal_patterns(df_clean)
            recent_activity = prepare_recent_activity(df_clean)
            prompt_filled = prompt.format(
                category_spending_data=category_data,
                temporal_patterns=temporal_data,
                cluster_summary=cluster_summary.to_string(),
                recent_activity=recent_activity,
                goals=goals,
                income=income,
                timeframe=timeframe,
                target_amount=target_amount
            )
            try:
                model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
                response = model.generate_content(prompt_filled)
                budget_response = response.text.strip()
            except Exception as e:
                budget_response = f"[ERROR] Gemini failed: {e}"
        budget_table, total_budget, summary, structured_summary = parse_budget_json_and_summary(budget_response)
        if budget_table:
            df_budget = pd.DataFrame(budget_table)
            st.markdown("<h4>Your Gemini-Powered Budget Table:</h4>", unsafe_allow_html=True)
            st.dataframe(df_budget)
            if total_budget is not None:
                st.success(f"**Total Budget Allocation:** {total_budget:,.2f}")
            pdf_bytes = budget_table_to_pdf(df_budget)
            b64 = base64.b64encode(pdf_bytes.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="budget_table.pdf" style="text-decoration:none;">🗎 Download as PDF</a>'
            st.markdown(href, unsafe_allow_html=True)
            if show_pie:
                if 'allocation' in df_budget.columns and 'category' in df_budget.columns:
                    fig, ax = plt.subplots()
                    ax.pie(df_budget['allocation'], labels=df_budget['category'], autopct='%1.1f%%', startangle=140)
                    ax.set_title('Budget Allocations by Category')
                    st.pyplot(fig)
                else:
                    st.warning("Pie chart requires 'allocation' and 'category' columns in the budget table.")
            # --- Structured summary UI ---
            if structured_summary:
                st.markdown("<h4>Budget Summary</h4>", unsafe_allow_html=True)
                st.info(f"**Key Savings Opportunities:** {structured_summary['Key Savings Opportunities']}")
                st.info(f"**Biggest Spending Categories:** {structured_summary['Biggest Spending Categories']}")
                st.warning(f"**Warnings/Advice:** {structured_summary['Warnings/Advice']}")
            elif summary:
                st.markdown(f"<div class='gemini-answer-box'>{summary}</div>", unsafe_allow_html=True)
            # --- Budget modification UI ---
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h4>✏️ Modify Your Budget</h4>", unsafe_allow_html=True)
            st.write("If you'd like to adjust your budget allocations or notes, edit the table below and click 'Save Changes'.")
            edited_df = st.data_editor(df_budget, num_rows="dynamic", use_container_width=True, key="budget_edit")
            if st.button("Save Changes"):
                st.success("Your budget changes have been saved! (You can now export or use this modified budget.)")
                # Optionally, you could re-calculate summary or allow re-running Gemini on the modified budget
        else:
            st.warning("Could not parse a valid budget table from Gemini's response.")
            if summary:
                st.markdown(f"<div class='gemini-answer-box'>{summary}</div>", unsafe_allow_html=True)




 # type: ignore