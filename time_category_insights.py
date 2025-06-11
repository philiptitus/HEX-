import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def time_category_insights_ui(df_clean):
    st.markdown("""
    <hr style='border: 1px solid #4F8BF9; margin-top: 30px; margin-bottom: 20px;'>
    <h2 style='color: #4F8BF9;'>🔎 Category Insights by Time</h2>
    <p style='font-size: 1.1em;'>Select a day of the week or hour of the day to see which categories dominate your spending for that period.</p>
    """, unsafe_allow_html=True)
    option = st.radio("Investigate by:", ["Day of Week", "Hour of Day"], horizontal=True)
    
    if option == "Day of Week":
        day = st.selectbox("Select Day of the Week:", [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        filtered = df_clean[df_clean['DayOfWeek'] == day]
        title = f"Top Categories for {day}"
    else:
        hour = st.selectbox("Select Hour of the Day (24h):", list(range(24)), format_func=lambda x: f"{x:02d}:00")

        filtered = df_clean[df_clean['Hour'] == hour]
        title = f"Top Categories for {hour:02d}:00"
    if filtered.empty:
        st.info("No transactions found for this selection.")
        return
    
    cat_amounts = filtered.groupby('Category')['Amount'].sum().sort_values(ascending=False)



    fig, ax = plt.subplots()
    sns.barplot(x=cat_amounts.index, y=cat_amounts.values, palette='viridis', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Category')
    ax.set_ylabel('Total Amount Spent')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
