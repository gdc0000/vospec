def visualize_results(result_df, categories):
    """
    Generates horizontal bar plots for the most characteristic words per category.
    """
    st.write("### 📊 Most Characteristic Words per Category")
    for cat in categories:
        subset = result_df[result_df['Category'] == cat]
        if subset.empty:
            st.write(f"No significant characteristic words found for category **{cat}**.")
            continue
        # Select top 10 based on absolute test value
        subset_sorted = subset.reindex(subset['Test Value'].abs().sort_values(ascending=False).index)
        top_subset = subset_sorted.head(10)

        fig = px.bar(
            top_subset,
            x="Test Value",
            y="Term",
            orientation='h',
            title=f"Top Characteristic Words for Category: {cat}",
            labels={"Test Value": "Test Value", "Term": "Word"},
            height=400
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
