import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import talib
import requests
from bs4 import BeautifulSoup
import os
import PySimpleGUI as sg
import threading
import queue
import time
from datetime import datetime, timedelta
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm
# 设置日志
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 設置Seaborn樣式
sns.set(style='whitegrid')

font_path = "/Users/tangjiahong/Dropbox/Pytorch/stock_env/TaipeiSansTCBeta-Regular.ttf"
if not os.path.exists(font_path):
    print(f"Font file not found: {font_path}")
    # 如果找不到字体文件，可以尝试使用系统默认字体
    font_path = None

if font_path:
    # 添加字体文件
    fm.fontManager.addfont(font_path)
    chinese_font = fm.FontProperties(fname=font_path)
else:
    # 如果找不到指定字体，使用系统默认中文字体
    chinese_font = fm.FontProperties(family='sans-serif')

# 设置 Matplotlib 的默认字体
# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 使用一个通用的中文字体
plt.rcParams['axes.unicode_minus'] = False

# 定義股票列表
STOCKS = ["2330.TW", "2317.TW", "2382.TW", "2303.TW", "1216.TW", "3711.TW", "1303.TW", "2002.TW", "1301.TW", "3231.TW",
          "3045.TW", "2542.TW", "2449.TW", "5388.TW", "2376.TW", "2603.TW", "3035.TW", "2356.TW", "2357.TW", "2383.TW",
          "2360.TW", "2454.TW", "6505.TW", "2412.TW", "2308.TW", "2881.TW", "3017.TW", "3006.TW", "2408.TW", "2344.TW",
          "8046.TW", "3037.TW", "2891.TW", "2882.TW", "2353.TW", "1513.TW", "8996.TW", "6282.TW", "1795.TW", "2388.TW"]

ETFS = ["00929.TW", "00712.TW", "00637L.TW", "2884.TW", "00650L.TW", "2886.TW", "2892.TW", "00893.TW", "00706L.TW", "2880.TW", 
        "00830.TW", "00900.TW", "0050.TW", "5880.TW", "00673R.TW", "00715L.TW", "00633L.TW", "2812.TW", "00885.TW", "00895.TW", 
        "00662.TW", "00888.TW", "00646.TW", "00903.TW", "00683L.TW", "00642U.TW", "00770.TW", "00762.TW", "0051.TW", "00851.TW", 
        "006203.TW", "00660.TW"]

# 全局變量
stock_data = {}
data_queue = queue.Queue()

def fetch_news(ticker):
    url = f"https://tw.stock.yahoo.com/q/h?s={ticker}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    news_items = soup.find_all('li', class_='js-stream-content')

    news_list = []
    for item in news_items[:20]:  # Get top 20 news
        title = item.find('h3').text if item.find('h3') else 'No title'
        link = item.find('a')['href'] if item.find('a') else 'No link'
        news_list.append({'Title': title, 'Link': link})
    
    return news_list

def fetch_institutional_trading(ticker):
    # 模擬機構交易數據
    dates = pd.date_range(end=pd.Timestamp('today'), periods=30)
    data = {
        'Date': dates,
        'Foreign_Investor_Buy': np.random.randint(100, 1000, size=30),
        'Foreign_Investor_Sell': np.random.randint(100, 1000, size=30),
        'Investment_Trust_Buy': np.random.randint(100, 1000, size=30),
        'Investment_Trust_Sell': np.random.randint(100, 1000, size=30),
        'Dealer_Buy': np.random.randint(100, 1000, size=30),
        'Dealer_Sell': np.random.randint(100, 1000, size=30)
    }
    df = pd.DataFrame(data)
    return df

def analyze_stock(ticker):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*4)  # 获取4年的数据
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            logging.warning(f"No data available for {ticker}")
            return None
        
        df = data.copy()
        df['Return'] = df['Close'].pct_change()

        # 計算技術指標
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['RSI'] = talib.RSI(df['Close'])
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
        df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['Close'])
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])

        # 計算額外勝率
        additional_win_rate = 0
        if df['Close'].iloc[-1] > df['Middle_BB'].iloc[-1] * 1.05:
            additional_win_rate += 10
        if df['Close'].iloc[-1] > df['MA20'].iloc[-1] * 1.05:
            additional_win_rate += 10
        if df['Close'].iloc[-1] > df['MA10'].iloc[-1] * 1.05:
            additional_win_rate += 10
        if df['Close'].iloc[-1] > df['Upper_BB'].iloc[-1] * 1.05:
            additional_win_rate += 10
        if df['RSI'].iloc[-1] > 50 and df['RSI'].iloc[-1] > df['RSI'].iloc[-2]:
            additional_win_rate += 30
        if df['MACD'].iloc[-1] > 1 and df['MACD'].iloc[-1] > df['MACD'].iloc[-2]:
            additional_win_rate += 30

        # Prophet預測
        prophet_df = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        if forecast['yhat'].iloc[-1] > forecast['yhat'].iloc[-2]:
            additional_win_rate += 20

        # 獲取機構交易數據
        institutional_trading_df = fetch_institutional_trading(ticker)
        institutional_trading_df['Net_Buy'] = (
            institutional_trading_df['Foreign_Investor_Buy'] - institutional_trading_df['Foreign_Investor_Sell'] +
            institutional_trading_df['Investment_Trust_Buy'] - institutional_trading_df['Investment_Trust_Sell'] +
            institutional_trading_df['Dealer_Buy'] - institutional_trading_df['Dealer_Sell']
        )
        if (institutional_trading_df['Net_Buy'].tail(3) > 0).all() and (institutional_trading_df['Net_Buy'].iloc[-1] > institutional_trading_df['Net_Buy'].iloc[-2]):
            additional_win_rate += 15

        # 計算各種收益率和勝率
        daily_return = df['Return'].mean() * 100
        weekly_return = df['Return'].resample('W').mean().mean() * 100
        one_month_return = df.loc[df.index[-1] - pd.DateOffset(months=1):, 'Return'].mean() * 100
        six_months_return = df.loc[df.index[-1] - pd.DateOffset(months=6):, 'Return'].mean() * 100
        one_year_return = df.loc[df.index[-1] - pd.DateOffset(years=1):, 'Return'].mean() * 100

        daily_win_rate = (df['Return'].apply(lambda x: x > 0).mean()) * 100
        weekly_win_rate = (df['Return'].resample('W').apply(lambda x: (x > 0).mean()) * 100).mean()
        monthly_win_rate = (df['Return'].resample('M').apply(lambda x: (x > 0).mean()) * 100).mean()
        six_months_win_rate = (df.loc[df.index[-1] - pd.DateOffset(months=6):, 'Return'].apply(lambda x: (x > 0)).mean() * 100) if not df.loc[df.index[-1] - pd.DateOffset(months=6):, 'Return'].empty else None
        one_year_win_rate = (df.loc[df.index[-1] - pd.DateOffset(years=1):, 'Return'].apply(lambda x: (x > 0)).mean() * 100) if not df.loc[df.index[-1] - pd.DateOffset(years=1):, 'Return'].empty else None

        logging.info(f"Successfully analyzed {ticker}")
        return {
            'ticker': ticker,
            'current_price': round(df['Close'].iloc[-1], 2),
            'daily_change': round(((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100, 2),
            'weekly_change': round(((df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]) * 100, 2),
            'MA10': round(df['MA10'].iloc[-1], 2),
            'MA20': round(df['MA20'].iloc[-1], 2),
            'RSI': round(df['RSI'].iloc[-1], 2),
            'MACD': round(df['MACD'].iloc[-1], 2),
            'additional_win_rate': round(additional_win_rate, 2),
            'daily_return': round(daily_return, 2),
            'weekly_return': round(weekly_return, 2),
            'one_month_return': round(one_month_return, 2),
            'six_months_return': round(six_months_return, 2),
            'one_year_return': round(one_year_return, 2),
            'daily_win_rate': round(daily_win_rate, 2),
            'weekly_win_rate': round(weekly_win_rate, 2),
            'monthly_win_rate': round(monthly_win_rate, 2),
            'six_months_win_rate': round(six_months_win_rate, 2) if six_months_win_rate is not None else None,
            'one_year_win_rate': round(one_year_win_rate, 2) if one_year_win_rate is not None else None,
            'volume': int(df['Volume'].iloc[-1] / 1000),  # 转换为张数,
            'historical_data': df,
            'institutional_trading': institutional_trading_df,
            'forecast': forecast
        }
        
    except Exception as e:
        logging.error(f"Error processing {ticker}: {str(e)}")
        return None

def update_stock_data(tickers, window):
    while True:
        for ticker in tickers:
            try:
                result = analyze_stock(ticker)
                if result is not None:
                    stock_data[ticker] = result
                    window.write_event_value('-STOCK-UPDATED-', ticker)
                    logging.info(f"Updated data for {ticker}")
                else:
                    logging.warning(f"Failed to get data for {ticker}")
            except Exception as e:
                logging.error(f"Error processing {ticker}: {str(e)}")
        time.sleep(300)  # 每5分鐘更新一次
def create_chart(ticker, chart_type='price'):
    try:
        data = stock_data.get(ticker)
        if data is None:
            return create_error_chart("No data available")
        
        df = data['historical_data']
        if df.empty:
            return create_error_chart("Empty dataset")
        
        plt.close('all')  # 关闭所有现有图表
        fig, ax = plt.subplots(figsize=(16, 9))
        ax = fig.add_subplot(111)
        
        if chart_type == 'price':
            ax.plot(df.index, df['Close'], label='收盤價')
            ax.plot(df.index, df['MA10'], label='MA10')
            ax.plot(df.index, df['MA20'], label='MA20')
            ax.set_title(f'{ticker} 股價走勢')
            ax.set_xlabel('日期')
            ax.set_ylabel('價格')
            ax.legend()
        elif chart_type == 'volume':
            volume_in_lots = df['Volume'] / 1000
            ax.bar(df.index, volume_in_lots)
            ax.set_title(f'{ticker} 成交量 (張)')
            ax.set_xlabel('日期')
            ax.set_ylabel('成交量 (張)')
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        elif chart_type == 'macd':
            ax.plot(df.index, df['MACD'], label='MACD')
            ax.plot(df.index, df['MACD_signal'], label='Signal Line')
            ax.bar(df.index, df['MACD_hist'])
            ax.set_title(f'{ticker} MACD')
            ax.set_xlabel('日期')
            ax.set_ylabel('MACD')
            ax.legend()
        elif chart_type == 'rsi':
            ax.plot(df.index, df['RSI'])
            ax.axhline(y=70, color='r', linestyle='--')
            ax.axhline(y=30, color='g', linestyle='--')
            ax.set_title(f'{ticker} RSI')
            ax.set_xlabel('日期')
            ax.set_ylabel('RSI')
        elif chart_type == 'bollinger':
            ax.plot(df.index, df['Close'], label='收盤價')
            ax.plot(df.index, df['Upper_BB'], label='Upper BB')
            ax.plot(df.index, df['Middle_BB'], label='Middle BB')
            ax.plot(df.index, df['Lower_BB'], label='Lower BB')
            ax.set_title(f'{ticker} 布林通道')
            ax.set_xlabel('日期')
            ax.set_ylabel('價格')
            ax.legend()
        elif chart_type == 'institutional':
            institutional_df = data['institutional_trading']
            ax.bar(institutional_df['Date'], institutional_df['Net_Buy'])
            ax.set_title(f'{ticker} 三大法人買賣超')
            ax.set_xlabel('日期')
            ax.set_ylabel('買賣超量')
        else:
            return create_error_chart("Unknown chart type")
        
        plt.tight_layout()
        return fig
    except Exception as e:
        logging.error(f"Error creating chart: {str(e)}")
        return create_error_chart(str(e))

def create_error_chart(error_message):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.text(0.5, 0.5, f"Error: {error_message}", ha='center', va='center')
    ax.set_axis_off()
    return fig


def create_gui():
    sg.theme('LightBlue2')

    font_path = "/Users/tangjiahong/Dropbox/Pytorch/stock_env/TaipeiSansTCBeta-Regular.ttf"
    sg.set_options(font=(font_path, 12))

    layout = [
        [sg.Text('股票分析器', font=(font_path, 24), justification='center', expand_x=True)],
        [sg.Combo(['Stocks', 'ETFs'], default_value='Stocks', key='-ASSET_TYPE-', enable_events=True),
         sg.Button('刷新', font=(font_path, 12)), sg.Button('退出', font=(font_path, 12))],
        [sg.Table(values=[], headings=['代碼', '當前價格', '漲跌幅(日)', '漲跌幅(週)', 'MA10', 'MA20', 'RSI', 'MACD', '額外勝率', '成交量'],
                  auto_size_columns=False, col_widths=[10, 10, 10, 10, 10, 10, 10, 10, 10, 12],
                  justification='right', key='-TABLE-', enable_events=True, 
                  font=(font_path, 11), header_font=(font_path, 12, 'bold'))],
        [sg.Combo(['價格走勢', '成交量', 'MACD', 'RSI', '布林通道', '三大法人買賣'], default_value='價格走勢', key='-CHART_TYPE-', enable_events=True)],
        [sg.Canvas(key='-CANVAS-', size=(2400, 1200))],
    ]

    window = sg.Window('股票分析器', layout, finalize=True, resizable=True, size=(2400, 1600))
    return window


def main():
    window = create_gui()
    current_tickers = STOCKS
    current_sort_column = None
    reverse_sort = False
    fig_canvas_agg = None
    table_data = []

    # 初始化一些數據
    for ticker in current_tickers[:5]:  # 只處理前5個股票作為初始數據
        try:
            result = analyze_stock(ticker)
            if result is not None:
                stock_data[ticker] = result
                logging.info(f"Initialized data for {ticker}")
        except Exception as e:
            logging.error(f"Error initializing {ticker}: {str(e)}")

    update_thread = threading.Thread(target=update_stock_data, args=(current_tickers, window), daemon=True)
    update_thread.start()

    while True:
        event, values = window.read(timeout=100)

        if event in (sg.WINDOW_CLOSED, '退出'):
            break

        if event == '-STOCK-UPDATED-':
            # 更新表格
            update_table(window, current_tickers, current_sort_column, reverse_sort)

        # ... 其他事件处理 ...
        if event == '-STOCK-UPDATED-' or event == '刷新':
            update_table(window, current_tickers, current_sort_column, reverse_sort)

        if event == '-ASSET_TYPE-':
            current_tickers = STOCKS if values['-ASSET_TYPE-'] == 'Stocks' else ETFS
            update_thread.join(timeout=0.1)
            update_thread = threading.Thread(target=update_stock_data, args=(current_tickers,), daemon=True)
            update_thread.start()

        if event == '刷新' or not data_queue.empty():
            table_data = [[ticker] + [stock_data[ticker].get(key, 'N/A') for key in ['current_price', 'daily_change', 'weekly_change', 'MA10', 'MA20', 'RSI', 'MACD', 'additional_win_rate', 'volume']]
                          for ticker in current_tickers if ticker in stock_data]
            
            if current_sort_column is not None:
                table_data.sort(key=lambda x: x[current_sort_column] if x[current_sort_column] != 'N/A' else -float('inf'), reverse=reverse_sort)
            
            window['-TABLE-'].update(values=table_data)
            logging.info(f"Updated table with {len(table_data)} rows")

        if event == '-TABLE-':
            if len(values['-TABLE-']) > 0:
                selected_row = values['-TABLE-'][0]
                selected_ticker = table_data[selected_row][0]
                chart_type = values['-CHART_TYPE-']
                fig = create_chart(selected_ticker, chart_type.lower().replace(' ', '_'))
                
                if fig_canvas_agg:
                    delete_figure_agg(fig_canvas_agg)
                fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
            elif isinstance(event, tuple) and event[0] == '-TABLE-' and event[2][0] == 'column':  # 点击表头
                clicked_column = event[2][1]
                if clicked_column == current_sort_column:
                    reverse_sort = not reverse_sort
                else:
                    current_sort_column = clicked_column
                    reverse_sort = False
                
                table_data.sort(key=lambda x: x[clicked_column] if x[clicked_column] != 'N/A' else -float('inf'), reverse=reverse_sort)
                window['-TABLE-'].update(values=table_data)

        if event == '-CHART_TYPE-':
            if len(values['-TABLE-']) > 0:
                selected_row = values['-TABLE-'][0]
                selected_ticker = table_data[selected_row][0]
                chart_type = values['-CHART_TYPE-']
                fig = create_chart(selected_ticker, chart_type.lower().replace(' ', '_'))
                
                if fig_canvas_agg:
                    delete_figure_agg(fig_canvas_agg)
                fig_canvas_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

    window.close()
def update_table(window, current_tickers, current_sort_column, reverse_sort):
    table_data = [[ticker] + [stock_data[ticker].get(key, 'N/A') for key in ['current_price', 'daily_change', 'weekly_change', 'MA10', 'MA20', 'RSI', 'MACD', 'additional_win_rate', 'volume']]
                  for ticker in current_tickers if ticker in stock_data]
    
    if current_sort_column is not None:
        table_data.sort(key=lambda x: x[current_sort_column] if x[current_sort_column] != 'N/A' else -float('inf'), reverse=reverse_sort)
    
    window['-TABLE-'].update(values=table_data)
    logging.info(f"Updated table with {len(table_data)} rows")
    
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().pack_forget()
    plt.close('all')

if __name__ == "__main__":
    main()