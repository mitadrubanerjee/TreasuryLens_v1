# app.py

import os
import json
import streamlit as st
import requests
from typing import List, Dict, Tuple
import plotly.express as px
from openai import OpenAI
from bs4 import BeautifulSoup  # Ensure it's uncommented in your local env
import feedparser
from datetime import datetime, timedelta

# ── Streamlit Config ───────────────────────────────────────
st.set_page_config(page_title="TreasuryLens", layout="wide", initial_sidebar_state="expanded")

# ── Global CSS Injection ───────────────────────────────────
st.markdown("""
            
<style>
h1, h2, h3 {
    text-align: center;
}
</style>
            
<style>
  .card {
    background-color: #1f2937;
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
  }
  .card h3 {
    margin-top: 0;
    color: #ffffff;
  }
  .card ul {
    padding-left: 1.2rem;
  }
  .card li {
    color: #e5e7eb;
    margin-bottom: 0.4rem;
  }

  .metric-positive {
    background-color: #cfe8fc;
    color: #1e3a8a;
    padding: 0.4rem 0.8rem;
    border-radius: 0.4rem;
    margin-bottom: 0.8rem;
    display: inline-block;
    font-size: 1rem;
  }

  .metric-neutral {
    background-color: #fceecf;
    color: #78350f;
    padding: 0.4rem 0.8rem;
    border-radius: 0.4rem;
    margin-bottom: 0.8rem;
    display: inline-block;
    font-size: 1rem;
  }

  .metric-negative {
    background-color: #fcdede;
    color: #7f1d1d;
    padding: 0.4rem 0.8rem;
    border-radius: 0.4rem;
    margin-bottom: 0.8rem;
    display: inline-block;
    font-size: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ── API Keys ───────────────────────────────────────────────
BING_KEY = st.secrets["bing"]["api_key"]
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/news/search"
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# ── Fetch Economic Calendar ────────────────────────────────

import requests
from datetime import datetime, timedelta
from typing import List, Dict

@st.cache_data(show_spinner=False)
def scrape_calendar() -> List[Dict]:
    api_key = st.secrets["tradingeconomics"]["api_key"]

    today = datetime.today()
    end_date = today + timedelta(days=4)

    url = "https://api.tradingeconomics.com/calendar"
    params = {
        "c": api_key,
        "country": "united states,eurozone,united kingdom,japan,china",
        # "importance": "2,3",  # Optional - comment this for now
        "start_date": today.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()

        try:
            data = r.json()
        except Exception:
            st.error("Received a malformed response (not JSON).")
            return []

        events = []
        for item in data:
            try:
                dt = datetime.strptime(item["Date"], "%Y-%m-%dT%H:%M:%S")
                events.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "weekday": dt.strftime("%a"),
                    "region": item.get("Country", "Unknown"),
                    "event": item.get("Category", "Event"),
                })
            except:
                continue

        return events

    except Exception as e:
        st.error(f"TradingEconomics API error: {e}")
        return []



# ── Bing Headline Fetchers ────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_global_headlines() -> List[str]:
    try:
        params = {"q": "forex market news", "count": 30, "mkt": "en-US", "safeSearch": "Off"}
        headers = {"Ocp-Apim-Subscription-Key": BING_KEY}
        r = requests.get(BING_ENDPOINT, params=params, headers=headers)
        r.raise_for_status()
        data = r.json().get("value", [])
        return [f"{a.get('name','')} — {a.get('description','')}" for a in data]
    except Exception as e:
        st.error(f"Error fetching global headlines: {e}")
        return []

@st.cache_data(show_spinner=False)
def fetch_currency_headlines(pair: str) -> List[str]:
    try:
        params = {"q": f"{pair} forex news", "count": 20, "mkt": "en-US", "safeSearch": "Off"}
        headers = {"Ocp-Apim-Subscription-Key": BING_KEY}
        r = requests.get(BING_ENDPOINT, params=params, headers=headers)
        r.raise_for_status()
        data = r.json().get("value", [])
        return [f"{a.get('name','')} — {a.get('description','')}" for a in data]
    except Exception as e:
        st.error(f"Error fetching headlines for {pair}: {e}")
        return []

# ── GPT Analysis ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def analyze_with_gpt(snippets: List[str]) -> Tuple[List[str], str, Dict[str,int]]:
    if not snippets:
        return [], "neutral", {"positive":0, "neutral":0, "negative":0}
    joined = "\n".join(f"- {s}" for s in snippets)
    prompt = f"""
You are an FX market analyst. Here are the latest news items:
{joined}

1) In 10 sentences, summarize the key themes and market drivers. Max 30 words per sentence.
2) State the overall tone from [positive, neutral, negative].
3) Provide sentiment counts.
4) Mention main themes.

Respond only in this exact JSON format:

{{
  "summary_points": ["...", "...", "..."],
  "overall_sentiment": "positive|neutral|negative",
  "counts": {{"positive": X, "neutral": Y, "negative": Z}}
}}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        text = resp.choices[0].message.content
        result = json.loads(text)
        bullets = result.get("summary_points", [])
        tone = result.get("overall_sentiment", "neutral")
        counts = result.get("counts", {})
        for k in ("positive", "neutral", "negative"):
            counts.setdefault(k, 0)
        return bullets, tone, counts
    except json.JSONDecodeError:
        st.error("Could not parse GPT output.")
        return [], "neutral", {"positive":0, "neutral":0, "negative":0}
    except Exception as e:
        st.error(f"GPT analysis failed: {e}")
        return [], "neutral", {"positive":0, "neutral":0, "negative":0}

# ── Renderer: Week Ahead Grid ─────────────────────────────
from collections import defaultdict
from datetime import datetime, timedelta

 # See what’s coming from RSS


def render_week_ahead_horizontal(events: List[Dict]):
    st.markdown("---")
    st.markdown("### 📅 Week Ahead (Global Events)")

    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    cols = st.columns(len(days))

    # Group events by weekday
    grouped = {day: [] for day in days}
    for ev in events:
        wd = ev.get("weekday", "")
        if wd in grouped:
            grouped[wd].append(f"**{ev['region']}:** {ev['event']}")

    # Render each column (day)
    for i, day in enumerate(days):
        with cols[i]:
            st.markdown(f"**{day}**")
            if grouped[day]:
                for item in grouped[day]:
                    st.markdown(f"- {item}")
            else:
                st.markdown("*No events*")





# ── Renderer: Sentiment Panels ───────────────────────────
def render_global_panel(bullets: List[str], overall: str, breakdown: Dict[str,int]):
    st.markdown("### Global FX Sentiment")
    cls = f"metric-{overall.lower()}"
    st.markdown(f"""<div class="{cls}"><strong>Overall Sentiment:</strong> {overall.title()}</div>""", unsafe_allow_html=True)
    bullet_html = "\n".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(f"""<div class="card"><h3>Key Takeaways</h3><ul>{bullet_html}</ul></div>""", unsafe_allow_html=True)
    fig = px.pie(names=list(breakdown.keys()), values=list(breakdown.values()), hole=0.4)
    fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color="white", width=2)))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

def render_currency_panel(bullets: List[str], overall: str, breakdown: Dict[str,int]):
    st.markdown("### Currency-Pair Deep Dive")
    cls = f"metric-{overall.lower()}"
    st.markdown(f"""<div class="{cls}"><strong>Overall Sentiment:</strong> {overall.title()}</div>""", unsafe_allow_html=True)
    bullet_html = "\n".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(f"""<div class="card"><h3>Highlights</h3><ul>{bullet_html}</ul></div>""", unsafe_allow_html=True)
    fig = px.pie(names=list(breakdown.keys()), values=list(breakdown.values()), hole=0.4)
    fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color="white", width=2)))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# ── Main App ──────────────────────────────────────────────
def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "summary_ready" not in st.session_state:
        st.session_state.summary_ready = False

    st.title("TreasuryLens")
    st.subheader("Currency Market Insights")

    if st.button("Fetch Global FX Sentiment"):
        with st.spinner("Fetching and analyzing global news..."):
            try:
                snippets = fetch_global_headlines()
                bullets, overall, counts = analyze_with_gpt(snippets)
                st.session_state["summary_data"] = {
                    "snippets": snippets,
                    "bullets": bullets,
                    "overall": overall,
                    "counts": counts,
                }
                st.session_state.summary_ready = True
                st.session_state.chat_history = []
            except Exception as e:
                st.error(f"Could not fetch and analyze global sentiment: {e}")
                st.session_state.summary_ready = False

    if st.session_state.summary_ready:
        bullets = st.session_state["summary_data"]["bullets"]
        overall = st.session_state["summary_data"]["overall"]
        counts = st.session_state["summary_data"]["counts"]

        render_global_panel(bullets, overall, counts)

        st.markdown("#### Ask a follow-up question")
        user_followup = st.text_input("Your question:", key="followup_input")

        if st.button("Submit Follow-Up"):
            if user_followup.strip():
                with st.spinner("Thinking..."):
                    st.session_state.chat_history.append({"role": "user", "content": user_followup})

                    if not any("summary_of_sentiment" in m.get("name", "") for m in st.session_state.chat_history):
                        summary_context = "\n".join(f"- {pt}" for pt in bullets)
                        st.session_state.chat_history.insert(0, {
                            "role": "user",
                            "content": f"Summary of recent FX sentiment:\n{summary_context}",
                            "name": "summary_of_sentiment"
                        })

                    system_msg = {
                        "role": "system",
                        "content": "You are a helpful FX market assistant. Be concise, insightful, and use macro/FX terminology when relevant."
                    }

                    messages = [system_msg] + [
                        {k: v for k, v in m.items() if k in ["role", "content"]} for m in st.session_state.chat_history
                    ]

                    try:
                        response = client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=messages,
                            temperature=0.4,
                        )
                        reply = response.choices[0].message.content.strip()
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"Follow-up failed: {e}")

        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("#### Conversation History")
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                elif msg["role"] == "assistant":
                    st.markdown(f"**GPT:** {msg['content']}")

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()

    st.markdown("---")

    pair = st.selectbox("Select Currency Pair to Analyze:", ["EUR/USD", "EUR/GBP", "EUR/JPY", "EUR/AUD", "EUR/CAD", "EUR/INR", "USD/CNH", "EUR/CHF", "EUR/NOK", "USD/BRL", "USD/ZAR", "USD/MXN","USD/IDR"])
    if st.button("Analyze This Pair"):
        with st.spinner(f"Analyzing sentiment for {pair}..."):
            try:
                snippets = fetch_currency_headlines(pair)
                bullets, overall, counts = analyze_with_gpt(snippets)
                render_currency_panel(bullets, overall, counts)
            except Exception as e:
                st.error(f"Could not fetch or analyze {pair}: {e}")
    
    events = scrape_calendar()
    render_week_ahead_horizontal(events)

if __name__ == "__main__":
    main()

