import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import logging
import datetime
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import time

def get_news(ticker):
    # Simulated news fetching function
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, features='xml')
    news_items = soup.findAll('item')
    
    news_list = []
    for item in news_items[:50]:  # Get top 20 news
        title = item.title.text
        description = item.description.text
        sentiment = TextBlob(title + " " + description).sentiment.polarity
        news_list.append({
            'title': title,
            'description': description,
            'sentiment': sentiment
        })
    
    return news_list

def analyze_news(news_list):
    if not news_list:
        logging.warning("No news found for this stock.")
        return 0  # Return 0 or other default value
    avg_sentiment = sum(news['sentiment'] for news in news_list) / len(news_list)
    return avg_sentiment

def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

def calculate_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    return df

def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(df):
    roll_max = df['Close'].rolling(window=252, min_periods=1).max()
    daily_drawdown = df['Close'] / roll_max - 1.0
    max_daily_drawdown = daily_drawdown.rolling(window=252, min_periods=1).min()
    return max_daily_drawdown.min()

def prepare_features(df):
    df = df.copy()  # Create a distinct copy
    df['Returns'] = df['Close'].pct_change()
    df['Target'] = np.where(df['Returns'].shift(-1) > 0, 1, 0)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI']
    X = df[features]
    y = df['Target']
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy, classification_report(y_test, y_pred)

def analyze_stock(ticker, start_date, end_date, industry_keywords):
    try:
        df = get_stock_data(ticker, start_date, end_date)
        df = calculate_technical_indicators(df)
        
        returns = df['Close'].pct_change().dropna()
        sharpe_ratio = calculate_sharpe_ratio(returns)
        max_drawdown = calculate_max_drawdown(df)
        
        X, y = prepare_features(df.dropna())
        model, scaler, accuracy, report = train_model(X, y)
        
        technical_score = 0
        if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1]:
            technical_score += 1
        if df['Close'].iloc[-1] > df['SMA_50'].iloc[-1]:
            technical_score += 1
        if df['RSI'].iloc[-1] > 50:
            technical_score += 1
        if df['RSI'].iloc[-1] < 30:
            technical_score -= 1
        if df['RSI'].iloc[-1] > 70:
            technical_score -= 1
        
        stock_news = get_news(ticker)
        industry_news = []
        for keyword in industry_keywords:
            industry_news.extend(get_news(keyword))
        
        stock_sentiment = analyze_news(stock_news)
        industry_sentiment = analyze_news(industry_news)
        
        avg_sentiment = (stock_sentiment + industry_sentiment) / 2
        
        return {
            'ticker': ticker,
            'technical_score': technical_score,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'model_accuracy': accuracy,
            'classification_report': report,
            'news_sentiment': avg_sentiment,
            'news': stock_news + industry_news
        }
    except Exception as e:
        logging.error(f"Error analyzing stock {ticker}: {str(e)}")
        return None

def generate_report(results, start_date, end_date):
    report = f"股市分析綜合報告 ({start_date} 到 {end_date})\n\n"
    
    # 1. 整體市場概況
    avg_sentiment = np.mean([result['news_sentiment'] for result in results])
    market_trend = "上漲" if avg_sentiment > 0 else "下跌"
    report += f"1. 整體市場概況\n"
    report += f"   - 市場趨勢: {market_trend}\n"
    report += f"   - 平均情感得分: {avg_sentiment:.2f}\n\n"
    
    # 2. 行業分析
    industries = {
        '半導體': ['2330.TW', '2303.TW', '2454.TW'],
        '電子零組件': ['2317.TW', '2354.TW', '2382.TW'],
        '金融': ['2882.TW', '2881.TW', '2891.TW'],
        '通訊網路': ['2412.TW', '3045.TW', '4904.TW']
    }
    
    report += "2. 行業分析\n"
    for industry, tickers in industries.items():
        industry_results = [r for r in results if r['ticker'] in tickers]
        avg_tech_score = np.mean([r['technical_score'] for r in industry_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in industry_results])
        best_stock = max(industry_results, key=lambda x: x['sharpe_ratio'])
        
        report += f"   {industry}行業:\n"
        report += f"   - 平均技術得分: {avg_tech_score:.2f}\n"
        report += f"   - 平均夏普比率: {avg_sharpe:.2f}\n"
        report += f"   - 表現最佳股票: {best_stock['ticker']}\n\n"
    
    # 3. 表現最佳的五支股票
    top_stocks = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)[:10]
    report += "3. 表現最佳的五支股票\n"
    for i, stock in enumerate(top_stocks, 1):
        report += f"   {i}. {stock['ticker']}: 夏普比率 {stock['sharpe_ratio']:.2f}, 技術得分 {stock['technical_score']}, 情感得分 {stock['news_sentiment']:.2f}\n"
    report += "\n"
    
    # 4. ETF分析
    etfs = ['0050.TW', '0056.TW', '00878.TW', '00881.TW']
    report += "4. ETF分析\n"
    for etf in etfs:
        etf_data = next((r for r in results if r['ticker'] == etf), None)
        if etf_data:
            report += f"   - {etf}: 技術得分 {etf_data['technical_score']}, 夏普比率 {etf_data['sharpe_ratio']:.2f}, 情感得分 {etf_data['news_sentiment']:.2f}\n"
    report += "\n"
    
     # 5. 個股分析 (前10支)
    report += "5. 個股分析 (前10支)\n"
    for stock in results[:10]:
        report += f"   - {stock['ticker']}: 技術得分 {stock['technical_score']}, 夏普比率 {stock['sharpe_ratio']:.2f}, 最大回撤 {stock['max_drawdown']*100:.2f}%, 情感得分 {stock['news_sentiment']:.2f}\n"
    report += "\n"
    
    # 6. 機器學習模型表現
    accuracies = [r['model_accuracy'] for r in results]
    report += "6. 機器學習模型表現\n"
    report += f"   - 平均準確率: {np.mean(accuracies):.2f}\n"
    report += f"   - 最高準確率: {max(accuracies):.2f}\n"
    report += f"   - 最低準確率: {min(accuracies):.2f}\n\n"
    
    # 7. 結論與建議
    report += "7. 結論與建議\n"
    report += f"   - 整體市場呈{market_trend}趨勢\n"
    report += "   - 多數股票的夏普比率為負，表明相對於無風險利率，它們的表現不佳\n"
    report += f"   - {top_stocks[0]['ticker']} 有最高的夏普比率，可能值得進一步研究\n"
    report += "   - ETF的表現反映了整體市場趨勢\n"
    report += "   - 機器學習模型的預測準確率中等，建議結合其他分析方法使用\n\n"
    report += "建議:\n"
    report += "1. 密切關注表現最佳的五支股票\n"
    report += "2. 對於長期投資，可考慮等待市場企穩後再進場\n"
    report += "3. 對於 ETF 投資者，建議關注市場整體走勢，可能需要調整投資策略\n"
    report += "4. 持續監控個股的技術指標和基本面，尋找潛在的投資機會\n\n"
    report += "請注意，這份報告基於歷史數據和技術分析，不應被視為投資建議。投資決策應結合個人風險承受能力和更全面的市場分析。"
    
    return report

def main():
    start_date = '2022-01-01'
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    tickers = ['2330.TW', '2317.TW', '2454.TW', '2412.TW', '2308.TW', 
               '2881.TW', '1303.TW', '2882.TW', '2303.TW', '2002.TW',
               '0050.TW', '0056.TW', '00878.TW', '00881.TW']
    
    industry_keywords = {
        '半導體': ['semiconductor', 'chip', '半導體'],
        '電子零組件': ['electronics', 'components', '電子零組件'],
        '金融': ['financial', 'bank', '金融'],
        '通訊網路': ['communication', 'network', '通訊網路']
    }
    
    results = []
    for ticker in tickers:
        logging.info(f"分析股票 {ticker}")
        result = analyze_stock(ticker, start_date, end_date, industry_keywords.get(ticker, []))
        if result:
            results.append(result)
        logging.info(f"{ticker} 分析完成")
        time.sleep(1)  # Avoid too frequent requests
    
    if results:
        report = generate_report(results, start_date, end_date)
        
        with open('stock_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logging.info("報告已生成並保存為 stock_analysis_report.txt")
    else:
        logging.error("沒有成功分析任何股票，無法生成報告。")
        
if __name__ == "__main__":
    main()
