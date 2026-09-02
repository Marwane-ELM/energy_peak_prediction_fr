import re
from html import escape

import altair
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# Must be the first Streamlit call.
st.set_page_config(
    layout="wide",
    page_title="PikElek AI",
    page_icon="⚡"
)

# -------------------------------------------------------------------------
# Remove Streamlit's default header, toolbar, menu and top gap
# -------------------------------------------------------------------------

st.markdown(
    """
    <style>
        /*
        Remove Streamlit's complete top header.
        */
        [data-testid="stHeader"] {
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
        }

        /*
        Remove the toolbar containing Deploy, Rerun and other buttons.
        */
        [data-testid="stToolbar"] {
            display: none !important;
        }

        /*
        Remove the Deploy button if Streamlit renders it separately.
        */
        [data-testid="stAppDeployButton"] {
            display: none !important;
        }

        /*
        Remove the three-dot Streamlit menu.
        */
        #MainMenu {
            display: none !important;
            visibility: hidden !important;
        }

        /*
        Remove Streamlit's colored decoration line.
        */
        [data-testid="stDecoration"] {
            display: none !important;
        }

        /*
        Remove Streamlit's running/status indicator from the header.
        */
        [data-testid="stStatusWidget"] {
            display: none !important;
        }

        /*
        Reduce the empty space left after removing the header.

        Increase this value if your logo becomes too close to the top.
        For example: 1.5rem or 2rem.
        */
        [data-testid="stMainBlockContainer"],
        .block-container {
            padding-top: 1rem !important;
        }

        /*
        Smaller top spacing on mobile screens.
        */
        @media (max-width: 600px) {
            [data-testid="stMainBlockContainer"],
            .block-container {
                padding-top: 0.75rem !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Automatic refresh every 20 seconds.
refresh_count = st_autorefresh(
    interval=20_000,
    key="forecast_auto_refresh"
)


# -------------------------------------------------------------------------
# Application header
# -------------------------------------------------------------------------

st.markdown(
    """
        <style>
            .pikelek-header {
                margin-top: -0.8rem;
                margin-bottom: 3.6rem;
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


# -------------------------------------------------------------------------
# Retrieve prediction data
# -------------------------------------------------------------------------

try:
    response = requests.get(
        "http://127.0.0.1:8000/predict",
        timeout=60
    )
    response.raise_for_status()
    api_data = response.json()

except requests.RequestException as error:
    st.error(
        f"Impossible to retrieve prediction data from the API: {error}"
    )
    st.stop()


# -------------------------------------------------------------------------
# Create prediction DataFrames
# -------------------------------------------------------------------------

historical = pd.DataFrame(
    api_data["historical"],
    columns=[
        "timestamp",
        "hist_consumption_mw"
    ]
)

predictions = pd.DataFrame(
    api_data["predictions"],
    columns=[
        "timestamp",
        "consumption_mw"
    ]
)


historical["timestamp"] = pd.to_datetime(
    historical["timestamp"]
)

predictions["timestamp"] = pd.to_datetime(
    predictions["timestamp"]
)


# All half-hour periods displayed on the chart.
full_hours = pd.date_range(
    start="00:00",
    end="23:30",
    freq="30min"
).strftime("%H:%M")


def prepare_hourly_series(
    dataframe,
    value_column,
    output_column
):
    """
    Creates a series containing one value for each half-hour period.

    If the API returns multiple rows for the same displayed time,
    the value with the most recent timestamp is retained.
    """
    dataframe = dataframe.copy()

    dataframe["hour"] = (
        dataframe["timestamp"]
        .dt.strftime("%H:%M")
    )

    dataframe = dataframe.sort_values(
        "timestamp"
    )

    dataframe = dataframe.drop_duplicates(
        subset="hour",
        keep="last"
    )

    return (
        dataframe
        .set_index("hour")[value_column]
        .rename(output_column)
    )


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


df = pd.concat(
    [
        historical_series,
        predictions_series
    ],
    axis=1
)


# Ensure that all half-hour periods exist.
df = df.reindex(full_hours)
df.index.name = "hour"


# =========================================================================
# CHART CONFIGURATION
# =========================================================================

# Vertical axis limits.
Y_MIN = 28000
Y_MAX = 59000

# Reduced chart height.
CHART_HEIGHT = 430

# Explicit horizontal grid-line positions.
# Change this list if you want more or fewer horizontal lines.
Y_GRID_VALUES = [
    28000,
    35000,
    42000,
    50000,
    58000
]

# Indicator positions adjusted for the smaller chart.
INDICATOR_TITLE_Y = 62000
INDICATOR_VALUE_Y = 60000


def format_consumption(value):
    """
    Formats a consumption value for display.
    """
    if pd.isna(value):
        return "Unavailable"

    return f"{value:,.0f} MW".replace(",", " ")


# -------------------------------------------------------------------------
# Prepare chart data
# -------------------------------------------------------------------------

hover_df = df.reset_index().copy()

hover_df["Historical value label"] = (
    hover_df["Historical data"]
    .apply(format_consumption)
)

hover_df["Prediction value label"] = (
    hover_df["Predicted values"]
    .apply(format_consumption)
)

hover_df["historical_indicator_hour"] = "00:30"
hover_df["prediction_indicator_hour"] = "05:00"


indicator_titles_df = pd.DataFrame(
    {
        "hour": [
            "00:30",
            "05:00"
        ],
        "label": [
            "● Historical Consumption Data",
            "● Consumption Forecasts"
        ],
        "indicator_type": [
            "Historical",
            "Prediction"
        ],
        "y": [
            INDICATOR_TITLE_Y,
            INDICATOR_TITLE_Y
        ]
    }
)


chart_df = (
    df.reset_index()
    .melt(
        id_vars="hour",
        var_name="Series",
        value_name="Consumption (MW)"
    )
    .dropna(
        subset=["Consumption (MW)"]
    )
)


# Last chronological prediction returned by the API.
if predictions.empty:
    last_prediction_df = pd.DataFrame(
        columns=[
            "hour",
            "Series",
            "Consumption (MW)"
        ]
    )

else:
    last_prediction_timestamp = predictions["timestamp"].max()

    last_prediction_hour = (
        last_prediction_timestamp
        .strftime("%H:%M")
    )

    last_prediction_df = chart_df[
        (
            chart_df["Series"]
            == "Predicted values"
        )
        & (
            chart_df["hour"]
            == last_prediction_hour
        )
    ].copy()


# -------------------------------------------------------------------------
# Chart colors
# -------------------------------------------------------------------------

series_colors = altair.Scale(
    domain=[
        "Historical data",
        "Predicted values"
    ],
    range=[
        "#0B3D91",
        "#F28E2B"
    ]
)


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


# -------------------------------------------------------------------------
# Hover selection
# -------------------------------------------------------------------------

hover = altair.selection_point(
    fields=["hour"],
    on="pointermove",
    clear="mouseout",
    toggle=False,
    empty=False
)


# -------------------------------------------------------------------------
# Shared chart encodings
# -------------------------------------------------------------------------

base = altair.Chart(
    chart_df
).encode(
    x=altair.X(
        "hour:N",
        title="Hour",
        scale=altair.Scale(
            domain=full_hours.tolist()
        ),
        axis=altair.Axis(
            labelAngle=-90,
            labelPadding=6,
            tickSize=4
        )
    ),
    y=altair.Y(
        "Consumption (MW):Q",
        title="",
        scale=altair.Scale(
            domain=[
                Y_MIN,
                Y_MAX
            ],
            nice=False
        ),
        axis=altair.Axis(
            # Only these values receive ticks and horizontal grid lines.
            values=Y_GRID_VALUES,
            format=",.0f",
            grid=True,
            gridColor="#FFFFFF",
            gridOpacity=0.12,
            gridWidth=1,
            tickColor="#7A8594",
            domainColor="#7A8594",
            labelColor="#AAB4C2",
            labelPadding=8
        )
    )
)


# -------------------------------------------------------------------------
# Historical area
# -------------------------------------------------------------------------

historical_area = (
    base
    .transform_filter(
        altair.datum.Series
        == "Historical data"
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


# -------------------------------------------------------------------------
# Historical and prediction lines
# -------------------------------------------------------------------------

lines = (
    base
    .mark_line(
        strokeWidth=2.5,
        clip=True
    )
    .encode(
        color=altair.Color(
            "Series:N",
            title=None,
            scale=series_colors,
            legend=None
        ),
        detail="Series:N"
    )
)


# -------------------------------------------------------------------------
# Last prediction marker
# -------------------------------------------------------------------------

last_prediction_outer_ring = (
    altair.Chart(
        last_prediction_df
    )
    .mark_circle(
        size=320,
        color="#FFD6AD",
        filled=False,
        strokeWidth=3
    )
    .encode(
        x=altair.X(
            "hour:N",
            scale=altair.Scale(
                domain=full_hours.tolist()
            )
        ),
        y=altair.Y(
            "Consumption (MW):Q",
            scale=altair.Scale(
                domain=[
                    Y_MIN,
                    Y_MAX
                ],
                nice=False
            )
        )
    )
)


last_prediction_inner_dot = (
    altair.Chart(
        last_prediction_df
    )
    .mark_circle(
        size=100,
        color="#F28E2B",
        filled=True
    )
    .encode(
        x=altair.X(
            "hour:N",
            scale=altair.Scale(
                domain=full_hours.tolist()
            )
        ),
        y=altair.Y(
            "Consumption (MW):Q",
            scale=altair.Scale(
                domain=[
                    Y_MIN,
                    Y_MAX
                ],
                nice=False
            )
        )
    )
)


# -------------------------------------------------------------------------
# Hover rule
# -------------------------------------------------------------------------

vertical_rule = (
    altair.Chart(
        hover_df
    )
    .mark_rule(
        color="#FFFFFF",
        strokeDash=[5, 4],
        strokeWidth=1.5
    )
    .encode(
        x=altair.X(
            "hour:N",
            scale=altair.Scale(
                domain=full_hours.tolist()
            )
        ),
        opacity=altair.condition(
            hover,
            altair.value(0.9),
            altair.value(0)
        )
    )
)


# -------------------------------------------------------------------------
# Hover points
# -------------------------------------------------------------------------

selected_points = (
    base
    .mark_circle(
        size=115,
        filled=True,
        stroke="white",
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
            altair.value(1),
            altair.value(0)
        )
    )
)


# -------------------------------------------------------------------------
# Indicator titles
# -------------------------------------------------------------------------

indicator_titles = (
    altair.Chart(
        indicator_titles_df
    )
    .mark_text(
        fontSize=13,
        fontWeight="bold",
        align="left",
        baseline="middle"
    )
    .encode(
        x=altair.X(
            "hour:N",
            scale=altair.Scale(
                domain=full_hours.tolist()
            )
        ),
        y=altair.Y(
            "y:Q",
            scale=altair.Scale(
                domain=[
                    Y_MIN,
                    Y_MAX
                ],
                nice=False
            )
        ),
        text=altair.Text(
            "label:N"
        ),
        color=altair.Color(
            "indicator_type:N",
            scale=altair.Scale(
                domain=[
                    "Historical",
                    "Prediction"
                ],
                range=[
                    "#2B61BA",
                    "#F28E2B"
                ]
            ),
            legend=None
        )
    )
)


# -------------------------------------------------------------------------
# Dynamic indicator values
# -------------------------------------------------------------------------

historical_indicator_value = (
    altair.Chart(
        hover_df
    )
    .transform_filter(
        hover
    )
    .mark_text(
        fontSize=14,
        color="#FFFFFF",
        align="left",
        baseline="middle"
    )
    .encode(
        x=altair.X(
            "historical_indicator_hour:N",
            scale=altair.Scale(
                domain=full_hours.tolist()
            )
        ),
        y=altair.Y(
            datum=INDICATOR_VALUE_Y,
            scale=altair.Scale(
                domain=[
                    Y_MIN,
                    Y_MAX
                ],
                nice=False
            )
        ),
        text=altair.Text(
            "Historical value label:N"
        )
    )
)


prediction_indicator_value = (
    altair.Chart(
        hover_df
    )
    .transform_filter(
        hover
    )
    .mark_text(
        fontSize=14,
        color="#FFFFFF",
        align="left",
        baseline="middle"
    )
    .encode(
        x=altair.X(
            "prediction_indicator_hour:N",
            scale=altair.Scale(
                domain=full_hours.tolist()
            )
        ),
        y=altair.Y(
            datum=INDICATOR_VALUE_Y,
            scale=altair.Scale(
                domain=[
                    Y_MIN,
                    Y_MAX
                ],
                nice=False
            )
        ),
        text=altair.Text(
            "Prediction value label:N"
        )
    )
)


# -------------------------------------------------------------------------
# Invisible mouse detection area
# -------------------------------------------------------------------------

mouse_detector = (
    altair.Chart(
        hover_df
    )
    .mark_rect(
        opacity=0.001
    )
    .encode(
        x=altair.X(
            "hour:N",
            scale=altair.Scale(
                domain=full_hours.tolist()
            )
        ),
        y=altair.Y(
            datum=Y_MAX,
            scale=altair.Scale(
                domain=[
                    Y_MIN,
                    Y_MAX
                ],
                nice=False
            )
        ),
        y2=altair.datum(Y_MIN),
        tooltip=[
            altair.Tooltip(
                "hour:N",
                title="Hour"
            ),
            altair.Tooltip(
                "Historical data:Q",
                title="Historical consumption (MW)",
                format=",.0f"
            ),
            altair.Tooltip(
                "Predicted values:Q",
                title="Predicted consumption (MW)",
                format=",.0f"
            )
        ]
    )
    .add_params(
        hover
    )
)


# -------------------------------------------------------------------------
# Assemble and display chart
# -------------------------------------------------------------------------

chart = (
    altair.layer(
        historical_area,
        lines,
        last_prediction_outer_ring,
        last_prediction_inner_dot,
        vertical_rule,
        selected_points,
        indicator_titles,
        historical_indicator_value,
        prediction_indicator_value,
        mouse_detector
    )
    .resolve_scale(
        color="independent"
    )
    .properties(
        height=CHART_HEIGHT
    )
    .configure_view(
        stroke=None
    )
)


st.altair_chart(
    chart,
    width="stretch"
)


# =========================================================================
# ELECTRICITY DEMAND INFORMATION
# =========================================================================

try:
    response2 = requests.get(
        "http://127.0.0.1:8000/demand",
        timeout=60
    )
    response2.raise_for_status()
    api_data2 = response2.json()

except requests.RequestException as error:
    st.error(
        f"Impossible to retrieve electricity demand data from the API: "
        f"{error}"
    )
    st.stop()


# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def get_api_values(values, expected_size):
    """
    Always returns the expected number of elements.
    """
    if not isinstance(
        values,
        (list, tuple)
    ):
        values = []

    values = list(values)

    values.extend(
        [None] * max(
            0,
            expected_size - len(values)
        )
    )

    return values[:expected_size]


def get_first_value(value, default=None):
    """
    Retrieves the first value when the API returns a list or tuple.
    Otherwise, returns the value directly.
    """
    if isinstance(
        value,
        (list, tuple)
    ):
        return (
            value[0]
            if len(value) > 0
            else default
        )

    return (
        value
        if value is not None
        else default
    )


def as_boolean(value):
    """
    Converts different boolean representations into a boolean.
    """
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "oui",
            "y"
        }

    if isinstance(
        value,
        (int, float)
    ):
        return value != 0

    return bool(value)


def as_number(value):
    """
    Converts a value to a float.
    """
    if value is None:
        return None

    try:
        normalized_value = (
            str(value)
            .strip()
            .replace(" ", "")
            .replace(",", ".")
        )

        return float(normalized_value)

    except (TypeError, ValueError):
        return None


def translate_level(value):
    """
    Translates common demand levels returned by the API into English.
    """
    if (
        value is None
        or str(value).strip() == ""
    ):
        return "Not available"

    original_value = str(value).strip()
    normalized_value = original_value.lower()

    translations = {
        "low": "Low",
        "faible": "Low",
        "bas": "Low",
        "basse": "Low",

        "medium": "Medium",
        "moderate": "Medium",
        "moyen": "Medium",
        "moyenne": "Medium",
        "modéré": "Medium",
        "modérée": "Medium",

        "high": "High",
        "élevé": "High",
        "élevée": "High",
        "haut": "High",
        "haute": "High",

        "very high": "Very high",
        "très élevé": "Very high",
        "très élevée": "Very high",

        "stable": "Stable",
        "constant": "Stable",
        "constante": "Stable"
    }

    translated_value = translations.get(
        normalized_value,
        original_value
    )

    return escape(
        translated_value
    )


def format_mw(value):
    """
    Formats an electricity consumption value in MW.
    """
    number = as_number(value)

    if number is None:
        return "—"

    return f"{number:,.0f} MW".replace(",", " ")


def format_percentage(value):
    """
    Formats a percentage with an explicit sign.
    """
    number = as_number(value)

    if number is None:
        return "—"

    sign = "+" if number >= 0 else "−"

    return f"{sign}{abs(number):.1f}%"


def format_slope_rate(value):
    """
    Formats the slope index.
    """
    number = as_number(value)

    if number is None:
        return "—"

    if number > 0:
        sign = "+"
    elif number < 0:
        sign = "−"
    else:
        sign = ""

    return f"{sign}{abs(number):,.2f}"


def format_forecast_datetime(value):
    """
    Separates the forecast time and date.
    """
    if (
        value is None
        or str(value).strip() == ""
    ):
        return "—", "Time unavailable"

    raw_value = str(value).strip()

    if re.fullmatch(
        r"\d{1,2}:\d{2}(?::\d{2})?",
        raw_value
    ):
        time_parts = raw_value.split(":")

        formatted_time = (
            f"{int(time_parts[0]):02d}:"
            f"{int(time_parts[1]):02d}"
        )

        return formatted_time, "Today"

    parsed_value = pd.to_datetime(
        value,
        errors="coerce"
    )

    if pd.isna(parsed_value):
        return (
            escape(raw_value),
            "Forecast time"
        )

    return (
        parsed_value.strftime("%H:%M"),
        parsed_value.strftime("%b %d, %Y")
    )


# -------------------------------------------------------------------------
# Read demand API data
# -------------------------------------------------------------------------

validation_value = get_first_value(
    api_data2.get("ready"),
    default=False
)

validation = as_boolean(
    validation_value
)


# -------------------------------------------------------------------------
# Card styles
# -------------------------------------------------------------------------

cards_css = """
<style>
    .pikelek-insights {
        width: 100%;
        margin-top: 1.5rem;
        margin-bottom: 2.5rem;

        font-family:
            Inter,
            ui-sans-serif,
            system-ui,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }

    .pikelek-insights *,
    .pikelek-insights *::before,
    .pikelek-insights *::after {
        box-sizing: border-box;
    }

    .pikelek-insights-heading {
        margin-bottom: 1.2rem;
    }

    .pikelek-insights-heading h2 {
        margin: 0;
        color: #F7FAFC;
        font-size: 1.4rem;
        font-weight: 800;
        line-height: 1.25;
        letter-spacing: -0.035rem;
    }

    .pikelek-insights-heading p {
        margin: 0.4rem 0 0;
        color: #7F8B9D;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    .pikelek-service-unavailable {
        display: flex;
        align-items: flex-start;
        gap: 0.85rem;

        margin-bottom: 1.2rem;
        padding: 0.95rem 1.05rem;

        color: #C7CFDA;

        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;

        background:
            linear-gradient(
                135deg,
                rgba(71, 85, 105, 0.32),
                rgba(30, 41, 59, 0.45)
            );

        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.025);
    }

    .pikelek-service-unavailable-icon {
        display: flex;
        flex: 0 0 auto;
        align-items: center;
        justify-content: center;

        width: 31px;
        height: 31px;

        color: #AAB4C2;
        font-size: 0.95rem;
        font-weight: 800;

        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 10px;
        background: rgba(71, 85, 105, 0.32);
    }

    .pikelek-service-unavailable-content {
        min-width: 0;
    }

    .pikelek-service-unavailable-title {
        margin: 0;
        color: #E2E8F0;
        font-size: 0.88rem;
        font-weight: 800;
    }

    .pikelek-service-unavailable-text {
        margin: 0.22rem 0 0;
        color: #94A3B8;
        font-size: 0.78rem;
        line-height: 1.45;
    }

    .pikelek-insights-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        width: 100%;
    }

    .pikelek-insight-card {
        --card-color: #8ACAFF;
        --card-border: rgba(138, 202, 255, 0.38);
        --card-glow: rgba(138, 202, 255, 0.12);

        position: relative;
        display: flex;
        flex-direction: column;

        min-width: 0;
        min-height: 290px;
        padding: 1.35rem;
        overflow: hidden;

        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;

        background:
            radial-gradient(
                circle at 100% 0%,
                var(--card-glow),
                transparent 42%
            ),
            linear-gradient(
                145deg,
                #171F2C 0%,
                #101620 100%
            );

        box-shadow:
            0 14px 35px rgba(0, 0, 0, 0.20),
            inset 0 1px 0 rgba(255, 255, 255, 0.035);

        transition:
            transform 180ms ease,
            border-color 180ms ease,
            box-shadow 180ms ease;
    }

    .pikelek-insight-card:hover {
        transform: translateY(-4px);
        border-color: var(--card-border);

        box-shadow:
            0 18px 42px rgba(0, 0, 0, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.055);
    }

    .pikelek-insight-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 1.35rem;
        right: 1.35rem;

        height: 3px;

        border-radius: 0 0 5px 5px;
        background: var(--card-color);
        box-shadow: 0 0 16px var(--card-color);
    }

    .pikelek-insight-card.trend-up {
        --card-color: #F28E2B;
        --card-border: rgba(242, 142, 43, 0.46);
        --card-glow: rgba(242, 142, 43, 0.14);
    }

    .pikelek-insight-card.trend-down {
        --card-color: #53B5FF;
        --card-border: rgba(83, 181, 255, 0.46);
        --card-glow: rgba(83, 181, 255, 0.14);
    }

    .pikelek-insight-card.trend-stable,
    .pikelek-insight-card.trend-neutral {
        --card-color: #A78BFA;
        --card-border: rgba(167, 139, 250, 0.46);
        --card-glow: rgba(167, 139, 250, 0.14);
    }

    .pikelek-insight-card.spike-detected {
        --card-color: #FF6473;
        --card-border: rgba(255, 100, 115, 0.48);
        --card-glow: rgba(255, 100, 115, 0.14);
    }

    .pikelek-insight-card.spike-clear {
        --card-color: #44D7A8;
        --card-border: rgba(68, 215, 168, 0.44);
        --card-glow: rgba(68, 215, 168, 0.13);
    }

    .pikelek-insight-card.maximum-available {
        --card-color: #F4C95D;
        --card-border: rgba(244, 201, 93, 0.46);
        --card-glow: rgba(244, 201, 93, 0.14);
    }

    .pikelek-insight-card.maximum-unavailable {
        --card-color: #7F8B9D;
        --card-border: rgba(127, 139, 157, 0.40);
        --card-glow: rgba(127, 139, 157, 0.10);
    }

    .pikelek-insight-card.service-unavailable-card {
        --card-color: #667085;
        --card-border: rgba(148, 163, 184, 0.20);
        --card-glow: rgba(100, 116, 139, 0.08);

        cursor: not-allowed;
        border-color: rgba(148, 163, 184, 0.17);

        background:
            radial-gradient(
                circle at 100% 0%,
                rgba(100, 116, 139, 0.08),
                transparent 42%
            ),
            linear-gradient(
                145deg,
                #161D27 0%,
                #10151D 100%
            );

        box-shadow:
            0 10px 25px rgba(0, 0, 0, 0.16),
            inset 0 1px 0 rgba(255, 255, 255, 0.02);
    }

    .pikelek-insight-card.service-unavailable-card:hover {
        transform: none;
        border-color: rgba(148, 163, 184, 0.17);

        box-shadow:
            0 10px 25px rgba(0, 0, 0, 0.16),
            inset 0 1px 0 rgba(255, 255, 255, 0.02);
    }

    .pikelek-insight-card.service-unavailable-card::before {
        background: #667085;
        box-shadow: none;
    }

    .pikelek-insight-card.service-unavailable-card .pikelek-card-title,
    .pikelek-insight-card.service-unavailable-card .pikelek-card-value,
    .pikelek-insight-card.service-unavailable-card .pikelek-card-info-value {
        color: #94A3B8;
    }

    .pikelek-insight-card.service-unavailable-card
    .pikelek-card-description,
    .pikelek-insight-card.service-unavailable-card
    .pikelek-card-eyebrow,
    .pikelek-insight-card.service-unavailable-card
    .pikelek-card-info-label {
        color: #667085;
    }

    .pikelek-card-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.8rem;
    }

    .pikelek-card-heading {
        min-width: 0;
    }

    .pikelek-card-eyebrow {
        margin-bottom: 0.45rem;
        color: #778397;
        font-size: 0.66rem;
        font-weight: 800;
        letter-spacing: 0.09rem;
        text-transform: uppercase;
    }

    .pikelek-card-title {
        margin: 0;
        color: #F7FAFC;
        font-size: 1.05rem;
        font-weight: 750;
        line-height: 1.35;
    }

    .pikelek-card-icon {
        display: flex;
        flex: 0 0 auto;
        align-items: center;
        justify-content: center;

        width: 44px;
        height: 44px;

        color: var(--card-color);
        font-size: 1.4rem;
        font-weight: 850;

        border: 1px solid var(--card-border);
        border-radius: 13px;
        background: var(--card-glow);
    }

    .pikelek-card-main {
        margin-top: 1.5rem;
    }

    .pikelek-card-value {
        overflow: hidden;
        color: #FFFFFF;
        font-size: clamp(1.55rem, 2.2vw, 2rem);
        font-weight: 850;
        line-height: 1.1;
        letter-spacing: -0.055rem;
        text-overflow: ellipsis;
    }

    .pikelek-card-description {
        min-height: 2.7rem;
        margin-top: 0.7rem;

        color: #8D99AA;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .pikelek-card-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;

        max-width: 100%;
        margin-top: 0.9rem;
        padding: 0.4rem 0.68rem;

        color: var(--card-color);
        font-size: 0.7rem;
        font-weight: 800;
        line-height: 1.2;

        border: 1px solid var(--card-border);
        border-radius: 999px;
        background: var(--card-glow);
    }

    .pikelek-card-badge-dot {
        width: 6px;
        height: 6px;
        flex: 0 0 auto;

        border-radius: 50%;
        background: var(--card-color);
        box-shadow: 0 0 8px var(--card-color);
    }

    .pikelek-card-footer {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 0.8rem;

        margin-top: auto;
        padding-top: 1.15rem;

        border-top: 1px solid rgba(255, 255, 255, 0.065);
    }

    .pikelek-card-info {
        min-width: 0;
    }

    .pikelek-card-info-right {
        min-width: 0;
        text-align: right;
    }

    .pikelek-card-info-label {
        margin-bottom: 0.25rem;
        color: #687487;
        font-size: 0.63rem;
        font-weight: 800;
        letter-spacing: 0.055rem;
        text-transform: uppercase;
    }

    .pikelek-card-info-value {
        color: #DDE5EF;
        font-size: 0.84rem;
        font-weight: 700;
        line-height: 1.35;
    }

    @media (max-width: 1050px) {
        .pikelek-insights-grid {
            grid-template-columns: 1fr;
        }

        .pikelek-insight-card {
            min-height: 250px;
        }
    }

    @media (max-width: 600px) {
        .pikelek-insight-card {
            padding: 1.1rem;
        }

        .pikelek-card-footer {
            align-items: flex-start;
            flex-direction: column;
        }

        .pikelek-card-info-right {
            text-align: left;
        }
    }
</style>
"""


# -------------------------------------------------------------------------
# Values when the demand analysis service is available
# -------------------------------------------------------------------------

if validation:
    highest_value_data = get_api_values(
        api_data2.get("highest_value"),
        4
    )

    slope_data = get_api_values(
        api_data2.get("slope"),
        2
    )

    spike_data = get_api_values(
        api_data2.get("spike"),
        3
    )

    bool_increase = as_boolean(
        highest_value_data[0]
    )

    y_max = highest_value_data[1]
    time_y_max = highest_value_data[2]
    percentage_increase = highest_value_data[3]

    slope_rate = slope_data[0]
    slope_level = slope_data[1]

    bool_spike = as_boolean(
        spike_data[0]
    )

    y_spike = spike_data[1]
    y_spike_time = spike_data[2]


    # Consumption trend.
    slope_number = as_number(
        slope_rate
    )

    if slope_number is None:
        trend_title = "Trend unavailable"
        trend_icon = "●"
        trend_class = "trend-neutral"
        trend_badge = "Incomplete data"

    elif slope_number > 0:
        trend_title = "Consumption increasing"
        trend_icon = "↗"
        trend_class = "trend-up"
        trend_badge = "Upward trend"

    elif slope_number < 0:
        trend_title = "Consumption decreasing"
        trend_icon = "↘"
        trend_class = "trend-down"
        trend_badge = "Downward trend"

    else:
        trend_title = "Consumption stable"
        trend_icon = "→"
        trend_class = "trend-stable"
        trend_badge = "Stable trend"

    slope_level_label = translate_level(
        slope_level
    )


    # Energy spike.
    if bool_spike:
        spike_title = "Energy spike detected"
        spike_class = "spike-detected"
        spike_badge = "Active alert"
        spike_value = format_mw(
            y_spike
        )

        spike_time, spike_date = format_forecast_datetime(
            y_spike_time
        )

        spike_description = (
            "An important temporary increase in electricity consumption "
            "is expected."
        )

    else:
        spike_title = "No energy spike detected"
        spike_class = "spike-clear"
        spike_badge = "Normal conditions"
        spike_value = "—"
        spike_time = "—"
        spike_date = "Normal"

        spike_description = (
            "No energy consumption spike is expected."
        )


    # Maximum forecast.
    if bool_increase:
        maximum_title = "Forecast maximum"
        maximum_class = "maximum-available"
        maximum_badge = "Highest point"

        maximum_value = format_mw(
            y_max
        )

        maximum_time, maximum_date = format_forecast_datetime(
            time_y_max
        )

        maximum_percentage = format_percentage(
            percentage_increase
        )

    else:
        maximum_title = "Maximum unavailable"
        maximum_class = "maximum-unavailable"
        maximum_badge = "Not detected"
        maximum_value = "—"
        maximum_time = "—"
        maximum_date = "Time unavailable"
        maximum_percentage = "—"

    service_message_html = ""


# -------------------------------------------------------------------------
# Values when the demand analysis service is unavailable
# -------------------------------------------------------------------------

else:
    trend_title = "Analysis unavailable"
    trend_icon = "—"
    trend_class = "service-unavailable-card"
    trend_badge = "Waiting for service"
    slope_level_label = "—"
    slope_rate = None

    spike_title = "Analysis unavailable"
    spike_class = "service-unavailable-card"
    spike_badge = "Waiting for service"
    spike_value = "—"
    spike_time = "—"
    spike_date = "—"

    spike_description = (
        "Peak detection is temporarily unavailable."
    )

    maximum_title = "Analysis unavailable"
    maximum_class = "service-unavailable-card"
    maximum_badge = "Waiting for service"
    maximum_value = "—"
    maximum_time = "—"
    maximum_date = "—"
    maximum_percentage = "—"

    service_message_html = """
    <div class="pikelek-service-unavailable">
        <div class="pikelek-service-unavailable-icon">
            ◌
        </div>

        <div class="pikelek-service-unavailable-content">
            <p class="pikelek-service-unavailable-title">
                Forecast analysis temporarily unavailable
            </p>

            <p class="pikelek-service-unavailable-text">
                The peak and consumption trend detection service is currently
                preparing data. This section will update automatically once
                the service becomes available.
            </p>
        </div>
    </div>
    """


# -------------------------------------------------------------------------
# Card HTML
# -------------------------------------------------------------------------

cards_html = f"""
<section class="pikelek-insights">

    <header class="pikelek-insights-heading">
        <h2>Analysis of Electricity Consumption Forecasts</h2>

        <p>
            Summary of the electricity consumption forecast for the next
            5 hours.
        </p>
    </header>

    {service_message_html}

    <div class="pikelek-insights-grid">

        <article class="pikelek-insight-card {trend_class}">
            <div class="pikelek-card-header">
                <div class="pikelek-card-heading">
                    <div class="pikelek-card-eyebrow">
                        Consumption trend
                    </div>

                    <h3 class="pikelek-card-title">
                        {escape(trend_title)}
                    </h3>
                </div>

                <div class="pikelek-card-icon">
                    {trend_icon}
                </div>
            </div>

            <div class="pikelek-card-main">
                <div class="pikelek-card-value">
                    {slope_level_label}
                </div>

                <div class="pikelek-card-description">
                    Estimated change in electricity demand over the next
                    five hours.
                </div>

                <div class="pikelek-card-badge">
                    <span class="pikelek-card-badge-dot"></span>
                    <span>{escape(trend_badge)}</span>
                </div>
            </div>

            <footer class="pikelek-card-footer">
                <div class="pikelek-card-info">
                    <div class="pikelek-card-info-label">
                        Forecast horizon
                    </div>

                    <div class="pikelek-card-info-value">
                        Next 5 hours
                    </div>
                </div>

                <div class="pikelek-card-info-right">
                    <div class="pikelek-card-info-label">
                        Slope index
                    </div>

                    <div class="pikelek-card-info-value">
                        {format_slope_rate(slope_rate)}
                    </div>
                </div>
            </footer>
        </article>


        <article class="pikelek-insight-card {spike_class}">
            <div class="pikelek-card-header">
                <div class="pikelek-card-heading">
                    <div class="pikelek-card-eyebrow">
                        Energy spike detection
                    </div>

                    <h3 class="pikelek-card-title">
                        {escape(spike_title)}
                    </h3>
                </div>

                <div class="pikelek-card-icon">
                    ⚡
                </div>
            </div>

            <div class="pikelek-card-main">
                <div class="pikelek-card-value">
                    {spike_value}
                </div>

                <div class="pikelek-card-description">
                    {escape(spike_description)}
                </div>

                <div class="pikelek-card-badge">
                    <span class="pikelek-card-badge-dot"></span>
                    <span>{escape(spike_badge)}</span>
                </div>
            </div>

            <footer class="pikelek-card-footer">
                <div class="pikelek-card-info">
                    <div class="pikelek-card-info-label">
                        Expected time
                    </div>

                    <div class="pikelek-card-info-value">
                        {spike_time}
                    </div>
                </div>

                <div class="pikelek-card-info-right">
                    <div class="pikelek-card-info-label">
                        Status
                    </div>

                    <div class="pikelek-card-info-value">
                        {spike_date}
                    </div>
                </div>
            </footer>
        </article>


        <article class="pikelek-insight-card {maximum_class}">
            <div class="pikelek-card-header">
                <div class="pikelek-card-heading">
                    <div class="pikelek-card-eyebrow">
                        Maximum forecast value
                    </div>

                    <h3 class="pikelek-card-title">
                        {escape(maximum_title)}
                    </h3>
                </div>

                <div class="pikelek-card-icon">
                    ◆
                </div>
            </div>

            <div class="pikelek-card-main">
                <div class="pikelek-card-value">
                    {maximum_value}
                </div>

                <div class="pikelek-card-description">
                    Highest electricity consumption value identified
                    in the available forecast.
                </div>

                <div class="pikelek-card-badge">
                    <span class="pikelek-card-badge-dot"></span>
                    <span>{escape(maximum_badge)}</span>
                </div>
            </div>

            <footer class="pikelek-card-footer">
                <div class="pikelek-card-info">
                    <div class="pikelek-card-info-label">
                        Expected at
                    </div>

                    <div class="pikelek-card-info-value">
                        {maximum_time} · {maximum_date}
                    </div>
                </div>

                <div class="pikelek-card-info-right">
                    <div class="pikelek-card-info-label">
                        Estimated increase
                    </div>

                    <div class="pikelek-card-info-value">
                        {maximum_percentage}
                    </div>
                </div>
            </footer>
        </article>

    </div>
</section>
"""


# -------------------------------------------------------------------------
# Render cards
# -------------------------------------------------------------------------

st.html(
    cards_css + cards_html
)