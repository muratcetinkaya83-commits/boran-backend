import json
import os

import anthropic
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Kurulum
# ---------------------------------------------------------------------------

app = FastAPI(title="Boran Analiz Motoru API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")  # opsiyonel, haber/sentiment icin
CLAUDE_MODEL = "claude-sonnet-5"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Coingecko'da yaygin kripto sembollerinin id karsiliklari.
# Kullanici baska bir coin sorarsa Claude symbol'u oldugu gibi gonderir,
# eslesme bulunamazsa CoinGecko search endpoint'i ile ikinci bir deneme yapilir.
COINGECKO_ID_MAP = {
    "BTC": "bitcoin", "BITCOIN": "bitcoin",
    "ETH": "ethereum", "ETHEREUM": "ethereum",
    "SOL": "solana", "SOLANA": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "TRX": "tron",
    "TON": "the-open-network",
    "DOT": "polkadot",
    "MATIC": "matic-network",
    "LINK": "chainlink",
    "SHIB": "shiba-inu",
}


# ---------------------------------------------------------------------------
# Teknik indikator hesaplama
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    for i in range(period, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Veri kaynaklari (tool implementasyonlari)
# ---------------------------------------------------------------------------

def fetch_stock_data(ticker: str) -> dict:
    """ticker orn: THYAO, THYAO.IS, AAPL, TSLA"""
    symbol = ticker.upper().strip()
    candidates = [symbol]
    if not symbol.endswith(".IS"):
        candidates.append(f"{symbol}.IS")

    last_error = None
    for sym in candidates:
        try:
            df = yf.Ticker(sym).history(period="3mo")
            if df.empty or len(df) < 50:
                last_error = "yeterli veri yok"
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = df["Close"]
            volume = df["Volume"]

            rsi_series = compute_rsi(close)
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()

            current_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])
            daily_change = round((current_price - prev_price) / prev_price * 100, 2)

            recent_vol = float(volume.iloc[-1])
            avg_vol = float(volume.iloc[-11:-1].mean())
            volume_status = "YUKSEK_HACIM" if recent_vol > avg_vol * 1.2 else "NORMAL_HACIM"

            return {
                "ticker": sym,
                "current_price": round(current_price, 2),
                "daily_change_pct": daily_change,
                "rsi_14": round(float(rsi_series.iloc[-1]), 2),
                "ema20": round(float(ema20.iloc[-1]), 2),
                "ema50": round(float(ema50.iloc[-1]), 2),
                "volume_status": volume_status,
            }
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            continue

    return {"error": f"'{ticker}' icin veri bulunamadi. Detay: {last_error}"}


def _resolve_coingecko_id(symbol: str) -> str | None:
    key = symbol.upper().strip()
    if key in COINGECKO_ID_MAP:
        return COINGECKO_ID_MAP[key]

    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/search",
            params={"query": symbol},
            timeout=10,
        )
        resp.raise_for_status()
        coins = resp.json().get("coins", [])
        if coins:
            return coins[0]["id"]
    except requests.RequestException:
        pass
    return None


def fetch_crypto_data(symbol: str) -> dict:
    coin_id = _resolve_coingecko_id(symbol)
    if not coin_id:
        return {"error": f"'{symbol}' coini bulunamadi."}

    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": coin_id,
                "vs_currencies": "usd,try",
                "include_24hr_change": "true",
                "include_market_cap": "true",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get(coin_id)
        if not data:
            return {"error": f"'{symbol}' icin fiyat verisi donmedi."}

        return {
            "coin": coin_id,
            "price_usd": data.get("usd"),
            "price_try": data.get("try"),
            "change_24h_pct": round(data.get("usd_24h_change", 0.0), 2),
            "market_cap_usd": data.get("usd_market_cap"),
        }
    except requests.RequestException as e:
        return {"error": str(e)}


def fetch_news_sentiment(ticker: str) -> dict:
    if not ALPHAVANTAGE_API_KEY:
        return {"note": "Haber/sentiment API anahtari (ALPHAVANTAGE_API_KEY) tanimli degil, bu adim atlandi."}

    try:
        resp = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": ticker.upper().replace(".IS", ""),
                "apikey": ALPHAVANTAGE_API_KEY,
                "limit": 10,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        feed = data.get("feed", [])[:5]
        if not feed:
            return {"note": "Bu sembol icin guncel haber bulunamadi (Alpha Vantage cogunlukla ABD hisseleri icin veri sagliyor)."}

        items = [
            {
                "title": f.get("title"),
                "sentiment_label": f.get("overall_sentiment_label"),
                "sentiment_score": f.get("overall_sentiment_score"),
                "source": f.get("source"),
            }
            for f in feed
        ]
        return {"news": items}
    except requests.RequestException as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Claude tool tanimlari
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "get_stock_data",
        "description": (
            "Bir hisse senedinin guncel fiyatini ve teknik indikatorlerini "
            "(RSI-14, EMA20, EMA50, hacim durumu) getirir. BIST hisseleri icin "
            "sembolu .IS uzantisi olmadan gonder (orn: THYAO, ASELS). ABD "
            "hisseleri icin standart sembolu gonder (orn: AAPL, TSLA, NVDA)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Hisse sembolu, orn: THYAO, AAPL"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_crypto_data",
        "description": (
            "Bir kripto paranin guncel USD/TRY fiyatini, 24 saatlik degisim "
            "yuzdesini ve piyasa degerini getirir."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Kripto sembolu veya adi, orn: BTC, bitcoin, ETH, SOL"}
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_news_sentiment",
        "description": (
            "Bir hisse senediyle ilgili son haberleri ve haber duygu (sentiment) "
            "skorlarini getirir. Sadece ABD hisseleri icin guvenilir sonuc verir, "
            "BIST hisseleri icin genelde veri bulunmaz."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "ABD hisse sembolu, orn: AAPL, TSLA"}
            },
            "required": ["ticker"],
        },
    },
]

TOOL_IMPLEMENTATIONS = {
    "get_stock_data": lambda inp: fetch_stock_data(inp["ticker"]),
    "get_crypto_data": lambda inp: fetch_crypto_data(inp["symbol"]),
    "get_news_sentiment": lambda inp: fetch_news_sentiment(inp["ticker"]),
}

SYSTEM_PROMPT = """Sen Boran adinda, Turkce konusan bir borsa ve kripto analiz asistanisin.
Kullaniciyla dogal, samimi ama profesyonel bir Turkce ile konusursun.

Kurallar:
- Kullanici bir hisse veya kripto para hakkinda soru sorduginda, elindeki tool'lari
  kullanarak guncel veriyi kendin cek. Hangi sembole bakman gerektigine kullanicinin
  mesajindan sen karar ver (orn: "Tesla" -> TSLA, "bitcoin" -> BTC, "Turk Hava Yollari" -> THYAO).
- Teknik verileri (RSI, EMA20/EMA50, hacim) ve varsa haber sentiment'ini birlikte
  degerlendirerek bilgilendirici bir yorum yap.
- ASLA "kesinlikle al" / "kesinlikle sat" gibi net emir kipinde talimat verme. Bunun yerine
  gostergelerin ne anlama geldigini, olasi senaryolari ve riskleri anlat; nihai karari
  kullaniciya birak. Sen bir yatirim danismani degilsin, bilgilendirici bir analiz aracisin.
- Veri bulunamazsa durumu acikca soyle, veri uydurma.
- Cevaplarini kisa ve okunakli tut, gerekirse madde isaretleri kullan. Asiri emoji kullanma.
"""


def run_tool(name: str, tool_input: dict) -> dict:
    impl = TOOL_IMPLEMENTATIONS.get(name)
    if not impl:
        return {"error": f"Bilinmeyen tool: {name}"}
    try:
        return impl(tool_input)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# API modelleri ve endpoint'ler
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Boran Analiz Motoru"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY tanimli degil. Render ortam degiskenlerine ekleyin.",
        )

    messages = [{"role": "user", "content": req.message}]

    for _ in range(5):  # tool-use dongusu icin ust limit
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as e:
            raise HTTPException(status_code=502, detail=f"Claude API hatasi: {str(e)}")

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return ChatResponse(reply=final_text or "Bir yanit uretemedim, tekrar dener misin?")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    raise HTTPException(status_code=504, detail="Islem tamamlanamadi (tool-use dongusu limiti asildi).")


# ---------------------------------------------------------------------------
# Debug/manuel test icin ham veri endpoint'leri (opsiyonel, Android kullanmiyor)
# ---------------------------------------------------------------------------

@app.get("/api/stock/{ticker}")
def get_stock_raw(ticker: str):
    result = fetch_stock_data(ticker)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/crypto/{symbol}")
def get_crypto_raw(symbol: str):
    result = fetch_crypto_data(symbol)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
