"""
Zomato chatbot for the Streamlit dashboard (NVIDIA API, OpenAI-compatible).

How it works
------------
1. The user asks a question in plain English.
2. The model decides what to calculate and calls the `run_pandas` tool
   with a small pandas snippet (it never sees all 7,403 rows).
3. The snippet runs on YOUR dataframe, the result goes back to the model,
   and the model explains the answer in simple language.

Setup
-----
    pip install openai
    # set your key as an environment variable (never put it in the code):
    #   Windows (PowerShell):  setx NVIDIA_API_KEY "your-key"   then restart the terminal
    # or create  .streamlit/secrets.toml  containing:
    #   NVIDIA_API_KEY = "your-key"
    # or just paste the key into the box in the Ask AI tab while the app runs.
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "deepseek-ai/deepseek-v4.1-flash"
MAX_TOOL_ROUNDS = 6
MAX_OUTPUT_CHARS = 4000

EXAMPLE_QUESTIONS = [
    "Which segment has the best average rating?",
    "Top 5 cities by number of restaurants",
    "Best rated Italian restaurants with over 200 votes",
    "How many Underperformers offer online delivery?",
    "Compare average rating of price range 1 vs 4",
]


# --------------------------------------------------
# SAFE PANDAS TOOL
# --------------------------------------------------

BLOCKED_WORDS = [
    "import", "__", "open(", "exec", "eval", "compile", "input(",
    "os.", "sys.", "subprocess", "globals", "locals", "getattr",
    "setattr", "delattr", "to_csv", "to_excel", "to_pickle",
    "read_csv", "read_", "system",
]

SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in [
        "len", "min", "max", "sum", "abs", "round", "sorted", "range",
        "list", "dict", "set", "tuple", "str", "int", "float", "bool",
        "enumerate", "zip", "any", "all", "print",
    ]
}


def run_pandas(code: str, df: pd.DataFrame) -> str:
    """Run a pandas snippet on a COPY of df. The snippet must set `result`."""
    lowered = code.lower()
    for word in BLOCKED_WORDS:
        if word in lowered:
            return f"Blocked: the code may not use '{word}'. Use only pandas/numpy on `df`."

    # one shared namespace so comprehensions/lambdas can see df
    namespace = {"__builtins__": SAFE_BUILTINS, "df": df.copy(), "pd": pd, "np": np}
    try:
        exec(code, namespace)
    except Exception as e:  # return the error so the model can fix its code
        return f"Error: {type(e).__name__}: {e}"

    if "result" not in namespace:
        return "No `result` variable was set. Assign your final answer to `result`."

    result = namespace["result"]
    if isinstance(result, pd.DataFrame):
        text = result.head(40).to_string()
        if len(result) > 40:
            text += f"\n... ({len(result)} rows total, showing first 40)"
    elif isinstance(result, pd.Series):
        text = result.head(40).to_string()
        if len(result) > 40:
            text += f"\n... ({len(result)} rows total, showing first 40)"
    else:
        text = str(result)

    return text[:MAX_OUTPUT_CHARS]


TOOL = {
    "type": "function",
    "function": {
        "name": "run_pandas",
        "description": (
            "Run a pandas snippet on the restaurant dataframe `df` (pandas as `pd`, "
            "numpy as `np`). Assign the final answer to a variable named `result`. "
            "Use it for any count, average, ranking, filter or comparison. "
            "No imports, no file access."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code using df. Must set `result`.",
                }
            },
            "required": ["code"],
        },
    },
}


def build_system_prompt(df: pd.DataFrame) -> str:
    columns = "\n".join(f"- {c} ({df[c].dtype})" for c in df.columns)
    segments = ", ".join(df["quality_segment"].dropna().unique().tolist())
    return f"""You are the assistant inside a "Zomato Restaurant Intelligence" dashboard.
You answer questions about a dataset of {len(df):,} Zomato restaurants using the run_pandas tool.

Columns in `df`:
{columns}

Facts about the data:
- rating_clean is the rating (0-5). votes is the number of votes; log_votes is log(votes).
- quality_segment comes from KMeans clustering on rating and votes. The segments are: {segments}.
  Top Performers = highest rating and many votes; Above Average = good rating, moderate votes;
  Average = mediocre rating and very few votes (so the rating is less reliable);
  Underperformers = lowest ratings.
- price_range is 1 (cheapest) to 4 (most expensive). average_cost_for_two is in LOCAL currencies
  (see the currency column), so never compare it across countries without mentioning this.
- has_online_delivery, has_table_booking, is_delivering_now are 0/1 flags.
- primary_cuisine is the first cuisine listed; cuisines has the full list (use str.contains for it).

Rules:
- Never guess numbers. Always use run_pandas and base your answer on its output.
- If a question is unrelated to this restaurant data, politely say you can only answer questions about it.
- Keep answers short and clear, in plain language. Use a small markdown table for rankings/comparisons.
- Round numbers sensibly. If a search returns nothing, say so and suggest a nearby question.
- If the user's wording is ambiguous (e.g. "best"), say how you interpreted it (e.g. highest rating with enough votes)."""


# --------------------------------------------------
# LLM CALL
# --------------------------------------------------

def ask_llm(client, history, question, df):
    """history = list of {"role", "content"} text messages from earlier turns."""
    system = build_system_prompt(df)
    messages = [{"role": "system", "content": system}] + history + [
        {"role": "user", "content": question}
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[TOOL],
            temperature=0.2,      # low = more accurate numbers
            max_tokens=2000,
            stream=False,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    output = run_pandas(args.get("code", ""), df)
                except Exception as e:
                    output = f"Error reading tool arguments: {e}"
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": output}
                )
        else:
            return (msg.content or "").strip() or "Sorry, I got an empty answer. Please try again."

    return "Sorry, I couldn't work that out. Try asking in a simpler way."


def get_api_key():
    key = os.environ.get("NVIDIA_API_KEY")
    if key:
        return key
    try:
        return st.secrets["NVIDIA_API_KEY"]
    except Exception:
        return None


# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

def render_chat_tab(full_df: pd.DataFrame, filtered_df: pd.DataFrame):
    st.subheader("💬 Ask the Restaurant Assistant")
    st.caption(
        "Ask anything about the restaurants, cities, cuisines, ratings or quality segments."
    )

    # The CSV's old performence_segment column has wrong labels, so hide it from the bot
    drop_cols = [c for c in ["performence_segment"] if c in full_df.columns]

    scope = st.radio(
        "Answer using",
        ["All restaurants", "Only my current filters"],
        horizontal=True,
        help="'Only my current filters' uses whatever you selected in the sidebar.",
    )
    data = (full_df if scope == "All restaurants" else filtered_df).drop(columns=drop_cols)
    st.caption(f"The assistant can see **{len(data):,}** restaurants.")

    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input(
            "Enter your NVIDIA API key to start",
            type="password",
            help="Create one at build.nvidia.com. It is only kept for this session.",
        )
    if not api_key:
        st.info("Add your API key above (or see the setup notes in chatbot.py) to enable the chat.")
        return

    try:
        import openai
        from openai import OpenAI
    except ImportError:
        st.error("Run `pip install openai` in your terminal, then restart the app.")
        return

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    top_l, top_r = st.columns([4, 1])
    with top_r:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    # example question buttons
    clicked = None
    st.write("**Try one:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, q in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True):
            clicked = q

    # show past messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed = st.chat_input("Ask about the restaurants...")
    question = typed or clicked

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analysing the data..."):
                try:
                    # send only the last few text turns to keep it cheap
                    history = st.session_state.chat_history[-8:]
                    answer = ask_llm(client, history, question, data)
                except openai.AuthenticationError:
                    answer = "That API key was rejected. Please check it and try again."
                except openai.RateLimitError:
                    answer = "Too many requests right now. Please wait a moment and retry."
                except Exception as e:
                    answer = f"Something went wrong: {e}"
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
