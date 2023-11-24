import dalex as dx
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Set up data

file_path = "stackoverflow_full.csv"
df = pd.read_csv(file_path)

df = df.drop(columns=["Unnamed: 0"])
df = df.drop(columns="HaveWorkedWith")

n = len(df)


# Function to be used
def get_employed_bias_barfig(bias):
    df_bias = df.groupby([bias, "Employed"]).size().reset_index(name='Number')
    df_bias.Employed = (df_bias.Employed == 1)
    fig = px.bar(df_bias, x=bias, y='Number', color="Employed", title=f"Number of employed by : {bias}",
                 color_discrete_sequence=["grey", "darkred"])
    return fig


def get_employed_bias_piefig(bias):
    df_bias = df.groupby([bias, "Employed"]).size().reset_index(name='Number')
    fig1 = px.pie(df_bias, values='Number', names=bias, title='Analysis among the population')
    df_bias = df[df.Employed == 1].groupby([bias, "Employed"]).size().reset_index(name='Number')
    fig2 = px.pie(df_bias, values='Number', names=bias, title='Analysis among the Employed')
    return fig1, fig2


def get_salary_bias_boxfig(bias):
    fig = px.box(df, x=bias, y="PreviousSalary")
    return fig


TO_DROP = ["Accessibility", "Country", "HaveWorkedWith"]
VAL_COLS = ["PreviousSalary", "YearsCode", "YearsCodePro", "ComputerSkills"]
TO_DUMMIES = ["Age", "EdLevel", "Gender", "MentalHealth", "MainBranch"]


def get_data_lin_regression(list_col, y_col):
    val_cols = list(set(VAL_COLS).intersection(list_col).difference(y_col))
    to_dummies = list(set(TO_DUMMIES).intersection(list_col).difference(y_col))
    if len(to_dummies) > 0:
        X = pd.get_dummies(df[to_dummies], drop_first=True, dtype=int)
    else:
        X = pd.DataFrame()
    X[val_cols] = df[val_cols]
    reg = LinearRegression().fit(X, df[y_col])

    result_df = pd.DataFrame({
        "Variables": reg.feature_names_in_,
        "Coeff.": reg.coef_
    })
    return result_df, reg.score(X, df[y_col])


def get_data_log_regression(list_col):
    val_cols = list(set(VAL_COLS).intersection(list_col))
    to_dummies = list(set(TO_DUMMIES).intersection(list_col))
    if len(to_dummies) > 0:
        X = pd.get_dummies(df[to_dummies], drop_first=True, dtype=int)
    else:
        X = pd.DataFrame()
    X[val_cols] = df[val_cols]
    reg = LogisticRegression(max_iter=10).fit(X, df["Employed"])

    prob = reg.predict_proba(X)[:, 0]
    delta_prob = []
    for key in reg.feature_names_in_:
        X_mod = X.copy()
        if key in VAL_COLS:
            X_mod[key] = X_mod[key] - 1
            prob_mod = reg.predict_proba(X_mod)[:, 0]
            delta_prob.append((prob_mod - prob).mean())
        else:  # To_dummies
            X_mod[key] = 0
            prob_mod = reg.predict_proba(X_mod)[:, 0]
            delta_prob.append((prob_mod - prob)[X[key] == 1].mean())

    result_df = pd.DataFrame({
        "Variables": reg.feature_names_in_,
        "Delta Prob.": [f"{round(x * 100, 2)} %" for x in delta_prob],
        "Coeff.": reg.coef_[0]
    })
    return result_df, reg.score(X, df["Employed"]), X, delta_prob


X = df.drop(columns='Employed')
y = df.Employed

categorical_features = ['Age', 'Accessibility', 'EdLevel', 'Gender', 'MentalHealth', 'MainBranch', 'Country']
numerical_features = ['YearsCode', 'YearsCodePro', 'PreviousSalary', 'ComputerSkills']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, categorical_features),
    ('num', 'passthrough', numerical_features)
])

explainer_dc = dx.Explainer.load(open("../explanations/dc_explanation.txt", 'rb'))

explainer_rf = dx.Explainer.load(open("../explanations/rf_explanation.txt", 'rb'))
explainer_rf_mitigated = dx.Explainer.load(open("../explanations/rf_m_explanation.txt", 'rb'))

explainer_qb = dx.Explainer.load(open("../explanations/gb_explanation.txt", 'rb'))
explainer_qb_mitigated = dx.Explainer.load(open("../explanations/gb_m_explanation.txt", 'rb'))


def get_fairness_check(criteria, privileged):
    protected = df[criteria]
    fobject1 = explainer_dc.model_fairness(protected=protected, privileged=privileged, label="Decision Tree")
    fobject2 = explainer_rf.model_fairness(protected=protected, privileged=privileged, label="Random Forest")
    fobject4 = explainer_qb.model_fairness(protected=protected, privileged=privileged, label="Gradient Boosting")
    return lambda t: fobject1.plot([fobject2, fobject4], type=t, show=False)


def get_fairness_comparison(criteria, privileged, model):
    protected = df[criteria]
    lookup = {"Random Forest": [explainer_rf, explainer_rf_mitigated],
              "Gradient Boosting": [explainer_qb, explainer_qb_mitigated]}
    fobject = lookup[model][0].model_fairness(protected=protected, privileged=privileged, label=model)
    fobject_mitigated = lookup[model][1].model_fairness(protected=protected, privileged=privileged,
                                                label=(model + " (Mitigated)"))
    return lambda t: fobject.plot([fobject_mitigated], type=t, show=False)


def save_data_unbias(result_df, score, X, list_col_correct, delta_prob):
    df_mod = df.copy()
    for key in list_col_correct:
        i = result_df.Variables.index(key)
        p = delta_prob[i]
        if key in VAL_COLS:
            df_mod["Employed"] -= df_mod[key] * p
        else:
            df_mod.loc[X[key] == 1, "Employed"] -= p
    rand = np.random.random(n)
    # A finir


# Sidebar
with st.sidebar:
    st.title('Ekimetrics Responsible AI')
    # st.subheader(f"File name: {filename}")
    st.image(['LOGO-ENSAE-BLANC.png', 'LOGO-ENSAE.png'][1])
    st.subheader("Alyette")
    st.subheader("Johanne")
    st.subheader("Jacques")
    st.subheader("Rémy")
    st.subheader("Théo")
    st.subheader("Tien-Thinh")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Analysis: employment",
    "Analysis: salary",
    "Linear Regression",
    "Logistic Regression",
    "Fairness test",
    "Bias Mitigation"
])

# Exploratory Data Analysis
## Employed population
tab1.header("Data analysis by employment")
bias1 = tab1.selectbox(
    "Which criteria to analyse with employment ?",
    ["Age", "Gender", "Country", "MentalHealth", "Accessibility"]
)

# tab1.subheader("The Dataset")
tab1.plotly_chart(get_employed_bias_barfig(bias1))

# tab1.subheader("The Employment")
fig1, fig2 = get_employed_bias_piefig(bias1)
tab1.plotly_chart(fig1)
tab1.plotly_chart(fig2)

## Salary Bias
tab2.header("Data analysis by Salary")
bias2 = tab2.selectbox(
    "Which criteria to analyse with salary ?",
    ["Age", "Gender", "Country", "MentalHealth", "Accessibility"]
)
tab2.plotly_chart(get_salary_bias_boxfig(bias2))

## Linear Regression
tab3.header("Linear Regression")
list_col3 = tab3.multiselect('Select variable for Linear Regress: ', VAL_COLS + TO_DUMMIES + ["Employed"],
                             default=VAL_COLS[1:] + TO_DUMMIES + ["Employed"])
y_col3 = tab3.selectbox('Select', VAL_COLS)
result_df3, score3 = get_data_lin_regression(list_col3, y_col3)
tab3.subheader(f'The score is: {round(score3 * 100, 2)}%')
tab3.table(result_df3)

## Logistic Regression
tab4.header("Logistic Regression")
list_col = tab4.multiselect('Select variable for Logistic Regress: ', VAL_COLS + TO_DUMMIES,
                            default=VAL_COLS[1:] + TO_DUMMIES)
result_df, score, X, delta_prob = get_data_log_regression(list_col=list_col)
tab4.subheader(f'The score is: {round(score * 100, 2)}%')
tab4.table(result_df)
tab4.subheader(f'Correcting Bias')
list_col_correct = tab4.multiselect('Select bias to correct: ', result_df.Variables, default=result_df.Variables)
# st.button("Unbias data", on_click=save_data_unbias, args=(result_df, score, X, delta_prob, list_col_correct))

## Fairness check


tab5.header("Models on biased dataset performance:")

tab5.subheader("Decision Tree performance")

tab5.table(explainer_dc.model_performance().result)

tab5.subheader("Random Forest performance")

tab5.table(explainer_rf.model_performance().result)

tab5.subheader("Gradient Boosting performance")

tab5.table(explainer_qb.model_performance().result)

tab5.header("Fairness check")

bias5_1 = tab5.selectbox(
    "Which criteria to check fairness on ?",
    ["Age", "Gender", "MentalHealth", "Accessibility"]
)

bias5_2 = tab5.selectbox(
    "Which value to be considered as \"privileged\" ?",
    set(df[bias5_1])
)

plot = get_fairness_check(bias5_1, bias5_2)

(t5_fairness_check,
 t5_metric_scores,
 t5_stacked,
 t5_radar,
 t5_performance_and_fairness,
 t5_heatmap) = tab5.tabs(
    ["Fairness Check",
     "Metric Scores",
     "Cumulated parity loss",
     "Radar",
     "Performance And Fairness",
     "Heatmap"
     ]
)

t5_fairness_check.plotly_chart(plot("fairness_check"), theme=None, use_container_width=True)
t5_metric_scores.plotly_chart(plot("metric_scores"), theme=None, use_container_width=True)
t5_stacked.plotly_chart(plot("stacked"), theme=None, use_container_width=True)
t5_radar.plotly_chart(plot("radar"), theme=None, use_container_width=True)
t5_performance_and_fairness.plotly_chart(plot("performance_and_fairness"), theme=None, use_container_width=True)
t5_heatmap.plotly_chart(plot("heatmap"), theme=None, use_container_width=True)

# tab5.plotly_chart(get_fairness_check(bias5_1, bias5_2))

## Bias Mitigation

tab6.header("Bias mitigation with Dalex:")

bias6_model = tab6.selectbox(
    "Which model should have its biases mitigated ?",
    # ["Decision Tree", "Random Forest", "kNN", "Gradient Boosting"],
    ["Random Forest", "Gradient Boosting"],
    key="bias6_model_selectbox"
)

bias6_1 = tab6.selectbox(
    "Which criteria to check fairness on ?",
    # ["Age", "Gender", "MentalHealth", "Accessibility"],
    ["Gender"],
    key="bias6_1_selectbox"
)

bias6_2 = tab6.selectbox(
    "Which value to be considered as \"privileged\" ?",
    # set(df[bias6_1]),
    ["Man"],
    key="bias6_2_selectbox"
)

plot = get_fairness_comparison(bias6_1, bias6_2, bias6_model)

(t6_fairness_check,
 t6_metric_scores,
 t6_stacked,
 t6_radar,
 t6_performance_and_fairness,
 t6_heatmap) = tab6.tabs(
    ["Fairness Check",
     "Metric Scores",
     "Cumulated parity loss",
     "Radar",
     "Performance And Fairness",
     "Heatmap"
     ]
)

t6_fairness_check.plotly_chart(plot("fairness_check"), theme=None, use_container_width=True)
t6_metric_scores.plotly_chart(plot("metric_scores"), theme=None, use_container_width=True)
t6_stacked.plotly_chart(plot("stacked"), theme=None, use_container_width=True)
t6_radar.plotly_chart(plot("radar"), theme=None, use_container_width=True)
t6_performance_and_fairness.plotly_chart(plot("performance_and_fairness"), theme=None, use_container_width=True)
t6_heatmap.plotly_chart(plot("heatmap"), theme=None, use_container_width=True)
