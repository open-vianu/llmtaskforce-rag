import altair as alt
import pandas as pd

source = pd.DataFrame({
    "Category":list("AAABBBCCC"),
    "Group":list("xyzxyzxyz"),
    "Value":[0.1, 0.6, 0.9, 0.7, 0.2, 1.1, 0.6, 0.1, 0.2]
})

alt.Chart(source).mark_bar().encode(
    x="Category:N",
    y="Value:Q",
    xOffset="Group:N",
    color="Group:N"
)