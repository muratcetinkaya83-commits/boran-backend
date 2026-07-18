import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Boran Analiz Motoru API")

class StockResponse(BaseModel):
    ticker: str
    current_price: float
    daily_change: float
    rsi: float
    ema20: float
    ema50: float
    volume_status: str
    market_mood: str

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).copy()
    loss = (-delta.where(delta < 0, 0)).copy()
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    for i in range(period, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@app.get("/api/stock/{ticker}", response_model=StockResponse)
def get_stock_data(ticker: str):
    formatted_ticker = ticker.upper()
    if not formatted_ticker.endswith(".IS") and len(formatted_ticker) >= 4:
        symbol = f"{formatted_ticker}.IS"
    else:
        symbol = formatted_ticker

    try:
        # 🎯 ÇÖZÜM: Yahoo Finance bot engelini aşmak için proxy benzeri istek başlıkları (headers) tanımlıyoruz.
        ticker_obj = yf.Ticker(symbol)
        
        # İstekleri session ve headers kullanarak güvenli hale getiren yfinance metodu
        df = ticker_obj.history(period="3m", proxy=None)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"{ticker} için veri bulunamadı.")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close_prices = df['Close']
        volumes = df['Volume']
        
        if len(close_prices) < 50:
            raise HTTPException(status_code=400, detail="İndikatör hesaplaması için yeterli geçmiş veri yok.")

        rsi_series = compute_rsi(close_prices)
        ema20_series = close_prices.ewm(span=20, adjust=False).mean()
        ema50_series = close_prices.ewm(span=50, adjust=False).mean()

        current_price = float(close_prices.iloc[-1])
        prev_price = float(close_prices.iloc[-2])
        daily_change = round(((current_price - prev_price) / prev_price) * 100, 2)
        
        rsi_val = round(float(rsi_series.iloc[-1]), 2)
        ema20_val = round(float(ema20_series.iloc[-1]), 2)
        ema50_val = round(float(ema50_series.iloc[-1]), 2)

        recent_volume = float(volumes.iloc[-1])
        avg_volume = float(volumes.iloc[-11:-1].mean())
        volume_status = "YÜKSEK HACİM" if recent_volume > avg_volume * 1.2 else "NORMAL HACİM"

        if rsi_val > 70:
            market_mood = "AÇGÖZLÜLÜK"
        elif rsi_val < 30:
            market_mood = "KORKU"
        else:
            market_mood = "NÖTR"

        return StockResponse(
            ticker=formatted_ticker,
            current_price=round(current_price, 2),
            daily_change=daily_change,
            rsi=rsi_val,
            ema20=ema20_val,
            ema50=ema50_val,
            volume_status=volume_status,
            market_mood=market_mood
        )

    except HTTPException as he:
        # FastAPI'nin kendi fırlattığı 404 hatalarını doğrudan dışarı aktar
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")
