import streamlit as st
import altair
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import requests


# Doit être le premier appel Streamlit.
st.set_page_config(
    layout="wide",
    page_title="PikElek AI",
    page_icon=":zap:"
)


# Rafraîchissement automatique toutes les 10 secondes.
refresh_count = st_autorefresh(
    interval=10_000,
    key="forecast_auto_refresh"
)

# Style et logo texte de l'application.
st.markdown(
    """
    <style>
        .pikelek-header {
            margin-top: -0.8rem;
            margin-bottom: 1.6rem;
            line-height: 1;
        }

        .pikelek-logo {
            color: #8acaff;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.06rem;
        }

        .pikelek-logo-ai {
            color: #F28E2B;
        }

        .pikelek-tagline {
            margin-top: 0.25rem;
            color: #7A8594;
            font-size: 0.68rem;
            font-weight: 500;
            letter-spacing: 0.04rem;
            text-transform: uppercase;
        }
    </style>

    <div class="pikelek-header">
        <div class="pikelek-logo">
            PikElek<span class="pikelek-logo-ai">.AI</span>
        </div>
        <div class="pikelek-tagline">
            Electricity Peak and Consumption Forecasts
        </div>
    </div>
    """,
    unsafe_allow_html=True
)




# Récupération des données depuis l'API FastAPI.
try:
    response = requests.get(
        "http://127.0.0.1:8000/predict",
        timeout=60
    )
    response.raise_for_status()
    api_data = response.json()

except requests.RequestException as error:
    st.error(f"Impossible de contacter l'API : {error}")
    st.stop()


# Création des DataFrames.
historical = pd.DataFrame(
    api_data["historical"],
    columns=["timestamp", "hist_consumption_mw"]
)

predictions = pd.DataFrame(
    api_data["predictions"],
    columns=["timestamp", "consumption_mw"]
)


# Conversion des timestamps.
historical["timestamp"] = pd.to_datetime(historical["timestamp"])
predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])


# Toutes les demi-heures de la journée affichées sur le graphique.
full_hours = pd.date_range(
    start="00:00",
    end="23:30",
    freq="30min"
).strftime("%H:%M")


def prepare_hourly_series(dataframe, value_column, output_column):
    """
    Crée une série contenant une valeur unique pour chaque heure.

    Si l'API retourne plusieurs lignes avec la même heure, par exemple
    plusieurs "14:30" pour différentes dates, seule la valeur du timestamp
    le plus récent est conservée.
    """
    dataframe = dataframe.copy()

    # Heure utilisée pour l'affichage : 00:00, 00:30, ..., 23:30.
    dataframe["hour"] = dataframe["timestamp"].dt.strftime("%H:%M")

    # Garantit que keep="last" correspond bien au timestamp le plus récent.
    dataframe = dataframe.sort_values("timestamp")

    # Corrige l'erreur :
    # ValueError: cannot reindex on an axis with duplicate labels
    dataframe = dataframe.drop_duplicates(
        subset="hour",
        keep="last"
    )

    # Crée une Series dont l'index est uniquement l'heure.
    return (
        dataframe
        .set_index("hour")[value_column]
        .rename(output_column)
    )


# Préparation des deux séries avec des index horaires uniques.
historical_series = prepare_hourly_series(
    dataframe=historical,
    value_column="hist_consumption_mw",
    output_column="Historical data"
)

predictions_series = prepare_hourly_series(
    dataframe=predictions,
    value_column="consumption_mw",
    output_column="Predicted values"
)


# Fusion de l'historique et des prédictions.
df = pd.concat(
    [
        historical_series,
        predictions_series
    ],
    axis=1
)


# S'assure que toutes les demi-heures de la journée existent dans le tableau.
df = df.reindex(full_hours)
df.index.name = "hour"



# ###################### SETTING UP THE GRAPH #########################
# Préparation du DataFrame pour Altair.
chart_df = (
    df.reset_index()
    .melt(
        id_vars="hour",
        var_name="Series",
        value_name="Consumption (MW)"
    )
    .dropna(subset=["Consumption (MW)"])
)

# Dernière prédiction chronologique retournée par l'API.
last_prediction_timestamp = predictions["timestamp"].max()
last_prediction_hour = last_prediction_timestamp.strftime("%H:%M")

# Récupère la valeur effectivement présente dans le graphique pour cette heure.
last_prediction_df = chart_df[
    (chart_df["Series"] == "Predicted values")
    & (chart_df["hour"] == last_prediction_hour)
].copy()


# Limites fixes de l'axe vertical.
Y_MIN = 28000
Y_MAX = 58000


# Couleurs des courbes.
series_colors = altair.Scale(
    domain=["Historical data", "Predicted values"],
    range=[
        "#0B3D91",  # Bleu historique.
        "#F28E2B"   # Orange prédiction.
    ]
)


# Dégradé de la zone historique.
historical_gradient = altair.Gradient(
    gradient="linear",
    x1=0,
    x2=0,
    y1=0,
    y2=1,
    stops=[
        altair.GradientStop(
            offset=0,
            color="#0B3D91"
        ),
        altair.GradientStop(
            offset=0.5,
            color="#4F8EDB"
        ),
        altair.GradientStop(
            offset=1,
            color="#D9ECFF"
        )
    ]
)


# Sélection de l'heure la plus proche de la souris.
hover = altair.selection_point(
    fields=["hour"],
    nearest=True,
    on="pointermove",
    clear="mouseout",
    empty=False
)


# Encodages communs du graphique.
base = altair.Chart(chart_df).encode(
    x=altair.X(
        "hour:N",
        title="Hour",
        scale=altair.Scale(domain=full_hours.tolist()),
        axis=altair.Axis(labelAngle=-90)
    ),
    y=altair.Y(
        "Consumption (MW):Q",
        title="Electricity consumption (MW)",
        scale=altair.Scale(
            domain=[Y_MIN, Y_MAX],
            nice=False
        )
    )
)


# Zone remplie sous la courbe historique.
historical_area = (
    base.transform_filter(
        altair.datum.Series == "Historical data"
    )
    .mark_area(
        color=historical_gradient,
        opacity=0.95,
        clip=True
    )
    .encode(
        y2=altair.datum(Y_MIN)
    )
)


# Courbes historique et prédite.
lines = (
    base.mark_line(
        strokeWidth=2.5,
        clip=True
    )
    .encode(
        color=altair.Color(
            "Series:N",
            title=None,
            scale=series_colors,
            legend=altair.Legend(
                orient="bottom",
                direction="horizontal"
            )
        ),
        detail="Series:N"
    )
)

# Anneau orange clair autour de la dernière prédiction.
last_prediction_outer_ring = (
    altair.Chart(last_prediction_df)
    .mark_circle(
        size=320,
        color="#FFD6AD",  # Orange clair.
        filled=False
    )
    .encode(
        x=altair.X(
            "hour:N",
            scale=altair.Scale(domain=full_hours.tolist())
        ),
        y=altair.Y(
            "Consumption (MW):Q",
            scale=altair.Scale(
                domain=[Y_MIN, Y_MAX],
                nice=False
            )
        )
    )
)


# Point orange central, de la même couleur que la courbe de prédiction.
last_prediction_inner_dot = (
    altair.Chart(last_prediction_df)
    .mark_circle(
        size=100,
        color="#F28E2B",  # Même orange que la courbe.
        filled=True
    )
    .encode(
        x=altair.X(
            "hour:N",
            scale=altair.Scale(domain=full_hours.tolist())
        ),
        y=altair.Y(
            "Consumption (MW):Q",
            scale=altair.Scale(
                domain=[Y_MIN, Y_MAX],
                nice=False
            )
        )
    )
)


# Ligne horizontale visible au survol.
horizontal_rules = (
    base.mark_rule(
        strokeDash=[5, 4],
        strokeWidth=1.5
    )
    .encode(
        color=altair.Color(
            "Series:N",
            scale=series_colors,
            legend=None
        ),
        opacity=altair.condition(
            hover,
            altair.value(0.85),
            altair.value(0)
        )
    )
)


# Point visible sur la donnée survolée.
selected_points = (
    base.mark_circle(
        size=100,
        filled=True
    )
    .encode(
        color=altair.Color(
            "Series:N",
            scale=series_colors,
            legend=None
        ),
        opacity=altair.condition(
            hover,
            altair.value(1),
            altair.value(0)
        ),
        tooltip=[
            altair.Tooltip(
                "hour:N",
                title="Hour"
            ),
            altair.Tooltip(
                "Series:N",
                title="Series"
            ),
            altair.Tooltip(
                "Consumption (MW):Q",
                title="Consumption (MW)",
                format=",.0f"
            )
        ]
    )
)


# Points invisibles utilisés pour détecter précisément la position de la souris.
mouse_detector = (
    base.mark_circle(
        size=180,
        opacity=0
    )
    .encode(
        tooltip=[
            altair.Tooltip(
                "hour:N",
                title="Hour"
            ),
            altair.Tooltip(
                "Series:N",
                title="Series"
            ),
            altair.Tooltip(
                "Consumption (MW):Q",
                title="Consumption (MW)",
                format=",.0f"
            )
        ]
    )
    .add_params(hover)
)


# Assemblage des couches du graphique.
chart = (
    altair.layer(
        historical_area,
        lines,
        last_prediction_outer_ring,
        last_prediction_inner_dot,
        horizontal_rules,
        selected_points,
        mouse_detector
    )
    .properties(
        height=450
    )
)


# Affichage du graphique.
st.altair_chart(
    chart,
    use_container_width=True
)


#################################################################





# ################ Tableaux de contrôle facultatifs. #######################
with st.expander("Afficher les données techniques"):
    st.subheader("Données historiques")
    st.dataframe(
        historical,
        use_container_width=True
    )

    st.subheader("Prédictions")
    st.dataframe(
        predictions,
        use_container_width=True
    )

    st.subheader("Données fusionnées utilisées par le graphique")
    st.dataframe(
        df,
        use_container_width=True
    )
# --------------------------------------------------------------------------