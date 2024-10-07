import pandas_ta as ta
import pandas as pd
import ccxt
import time
import config  # Файл, содержащий ваши API-ключи и секретные ключи
import logging
import asyncio
import csv
import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Настройка логирования для лучшей видимости работы бота
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("arbitrage_bot.log"),
        logging.StreamHandler()
    ]
)

# Создание экземпляров бирж
exchanges = {
    'binance': ccxt.binance({
        'apiKey': config.BINANCE_API_KEY,
        'secret': config.BINANCE_SECRET_KEY,
        'enableRateLimit': True,
    }),
    'kucoin': ccxt.kucoin({
        'apiKey': config.KUCOIN_API_KEY,
        'secret': config.KUCOIN_SECRET_KEY,
        'password': config.KUCOIN_PASSPHRASE,
        'enableRateLimit': True,
    })
}

# Параметры для арбитража
TRADE_AMOUNT = 0.001  # Количество BTC для торговли
SLIPPAGE_THRESHOLD = 0.5  # Максимальное проскальзывание в процентах
DRY_RUN = False  # Если True, бот не будет размещать реальные ордера (для тестирования)
MIN_BALANCE = 50  # Минимальный баланс в USDT для торговли (увеличен для соответствия минимальным требованиям бирж)
MIN_PROFIT = 1.5  # Минимальная прибыль в долларах для выполнения сделки
HISTORY_FILE = 'market_data.csv'  # Файл для хранения исторических данных
MODEL_FILE = 'lstm_price_predictor.h5'  # Файл для сохранения обученной модели
SCALER_FILE = 'scaler.pkl'  # Файл для сохранения скейлера данных
SUPPORTED_PAIRS = []  # Поддерживаемые пары для торговли (будет динамически обновляться)
MAX_RETRIES = 3  # Максимальное количество повторных попыток при ошибках подключения
STOP_LOSS_PERCENTAGE = 2.0  # Процент для Stop-Loss
TAKE_PROFIT_PERCENTAGE = 5.0  # Процент для Take-Profit

# Настройка ThreadPoolExecutor для многозадачности
executor = ThreadPoolExecutor(max_workers=20)  # Увеличили количество потоков для более эффективного использования CPU

async def run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

# Шаг 1: Скачивание и сохранение исторических данных
async def download_market_data():
    try:
        exchange = exchanges['binance']
        bars = await run_in_executor(exchange.fetch_ohlcv, 'BTC/USDT', '1h', 1000)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.to_csv(HISTORY_FILE, index=False)
        logging.info("Файл market_data.csv успешно создан и заполнен историческими данными.")
    except Exception as e:
        logging.error(f"Ошибка при скачивании и сохранении исторических данных: {e}")

# Шаг 2: Обучение модели и сохранение скейлера
async def train_lstm_model():
    try:
        df = pd.read_csv(HISTORY_FILE)
        data = df['close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        x_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            x_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=1, batch_size=32)
        model.save(MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)
        logging.info("Модель lstm_price_predictor.h5 и скейлер scaler.pkl успешно созданы и сохранены.")
    except Exception as e:
        logging.error(f"Ошибка при обучении модели LSTM: {e}")

# Основная функция для подготовки данных и модели
async def prepare_data_and_model():
    await download_market_data()
    await train_lstm_model()

# Функция для обновления поддерживаемых пар
async def update_supported_pairs():
    global SUPPORTED_PAIRS
    try:
        tasks = [run_in_executor(exchange.fetch_markets) for exchange in exchanges.values()]
        results = await asyncio.gather(*tasks)
        pairs = [set([market['symbol'] for market in result]) for result in results]
        SUPPORTED_PAIRS = list(set.intersection(*pairs))
        logging.info(f"Поддерживаемые пары обновлены: {SUPPORTED_PAIRS}")
    except Exception as e:
        logging.error(f"Ошибка при обновлении поддерживаемых пар: {e}")

# Функция для кеширования результатов анализа на длительный промежуток времени
market_analysis_cache = {}

async def analyze_market_conditions_with_cache(pair, timeframe='1h'):
    current_time = datetime.now()
    cache_key = f"{pair}_{timeframe}"

    # Проверяем, есть ли закешированный результат и не устарел ли он
    if cache_key in market_analysis_cache:
        cached_data, cache_time = market_analysis_cache[cache_key]
        if (current_time - cache_time).seconds < 1800:  # Кеширование на 30 минут
            return cached_data

    # Если кеш устарел или его нет, выполняем новый анализ
    result = await analyze_market_conditions(pair)
    market_analysis_cache[cache_key] = (result, current_time)
    return result

# Функция для анализа рыночных условий
async def analyze_market_conditions(pair):
    try:
        exchange = exchanges['binance']
        bars = await run_in_executor(exchange.fetch_ohlcv, pair, '1h', 1000)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.join(ta.macd(df['close'], fast=12, slow=26, signal=9))
        df = df.join(ta.bbands(df['close'], length=20))

        # Простейшая проверка условий на основе MACD и Bollinger Bands
        if df['MACD_12_26_9'].iloc[-1] > df['MACDs_12_26_9'].iloc[-1] and df['close'].iloc[-1] > df['BBL_20_2.0'].iloc[-1]:
            logging.info(f"Пара {pair} выглядит предпочтительной для арбитража.")
            return True
        return False
    except Exception as e:
        logging.error(f"Ошибка при анализе рыночных условий: {e}")
        return False

# Функция для анализа волатильности
async def analyze_volatility(pair):
    try:
        bars = await run_in_executor(exchanges['binance'].fetch_ohlcv, pair, '1h', 1000)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['volatility'] = (df['high'] - df['low']) / df['close'] * 100
        return df['volatility'].mean()
    except Exception as e:
        logging.error(f"Ошибка при анализе волатильности: {e}")
        return 0

# Функция для анализа объема торгов
async def analyze_trading_volume(pair):
    try:
        bars = await run_in_executor(exchanges['binance'].fetch_ohlcv, pair, '1h', 1000)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df['volume'].mean()
    except Exception as e:
        logging.error(f"Ошибка при анализе объема торгов: {e}")
        return 0

# Функция для анализа тренда
async def analyze_trend_strength(pair):
    try:
        bars = await run_in_executor(exchanges['binance'].fetch_ohlcv, pair, '1h', 1000)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['trend'] = df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        trend_strength = df['trend'].sum() / len(df)
        return trend_strength * 100
    except Exception as e:
        logging.error(f"Ошибка при анализе тренда: {e}")
        return 0

# Функция для установки динамических значений Stop-Loss и Take-Profit
async def set_dynamic_stop_loss_and_take_profit(entry_price, volatility, trading_volume, trend_strength):
    try:
        if entry_price is None:
            raise ValueError("Entry price is None")
        stop_loss_percentage = STOP_LOSS_PERCENTAGE * (1 + volatility / 100) * (1 - trend_strength / 100)
        take_profit_percentage = TAKE_PROFIT_PERCENTAGE * (1 + volatility / 100) * (1 + trading_volume / 100)
        stop_loss_price = entry_price * (1 - stop_loss_percentage / 100)
        take_profit_price = entry_price * (1 + take_profit_percentage / 100)
        return stop_loss_price, take_profit_price
    except Exception as e:
        logging.error(f"Ошибка при установке Stop-Loss и Take-Profit: {e}")
        return entry_price * 0.98 if entry_price else None, entry_price * 1.02 if entry_price else None  # Значения по умолчанию

# Функция для проверки баланса перед размещением ордера
async def check_balance(exchange, symbol, side, amount):
    try:
        balance = await run_in_executor(exchange.fetch_balance)
        base_currency = symbol.split('/')[0] if side == 'sell' else symbol.split('/')[1]
        available_balance = balance[base_currency]['free'] if base_currency in balance else 0
        return available_balance >= amount
    except Exception as e:
        logging.error(f"Ошибка при проверке баланса для пары {symbol}: {e}")
        return False

# Функция для размещения ордера
async def place_order(exchange, pair, side, amount, price=None):
    try:
        # Получение информации о минимальном количестве ордера для пары
        market = await run_in_executor(exchange.load_markets)
        min_amount = market[pair]['limits']['amount']['min'] if pair in market else 0.1
        min_notional = market[pair]['limits']['cost']['min'] if pair in market and 'cost' in market[pair]['limits'] else 10.0

        # Корректировка количества, если оно меньше минимального или не удовлетворяет минимальному номиналу
        if amount < min_amount:
            logging.warning(f"Корректировка количества для {pair} до минимального значения {min_amount}")
            amount = min_amount

        # Проверка, удовлетворяет ли сумма минимальному номиналу
        notional = amount * price if price else amount
        if notional < min_notional:
            logging.warning(f"Корректировка количества для {pair} до удовлетворения минимальному номиналу {min_notional}")
            amount = min_notional / price if price else min_notional

        # Проверка баланса перед размещением ордера
        has_balance = await check_balance(exchange, pair, side, amount)
        if not has_balance:
            logging.error(f"Недостаточно средств для размещения ордера для пары {pair}")
            return

        params = {'type': 'limit'} if price else {'type': 'market'}
        order = await run_in_executor(exchange.create_order, pair, 'limit' if price else 'market', side, amount, price, params)
        logging.info(f"Ордер размещен: {order}")
    except Exception as e:
        logging.error(f"Ошибка при размещении ордера для пары {pair}: {e}")

# Основная функция для арбитража
async def arbitrage_bot():
    await update_supported_pairs()
    while True:
        try:
            for pair in SUPPORTED_PAIRS:
                # Анализ рыночных условий и волатильности
                is_preferable = await analyze_market_conditions_with_cache(pair)
                if not is_preferable:
                    continue

                volatility = await analyze_volatility(pair)
                trading_volume = await analyze_trading_volume(pair)
                trend_strength = await analyze_trend_strength(pair)

                # Получение текущей цены актива
                ticker = await run_in_executor(exchanges['binance'].fetch_ticker, pair)
                entry_price = ticker['last']

                # Установка динамического Stop-Loss и Take-Profit
                stop_loss_price, take_profit_price = await set_dynamic_stop_loss_and_take_profit(entry_price, volatility, trading_volume, trend_strength)
                logging.info(f"Пара {pair}: Stop-Loss: {stop_loss_price}, Take-Profit: {take_profit_price}")

                # Размещение ордеров (покупка и продажа для арбитража)
                if not DRY_RUN:
                    await place_order(exchanges['binance'], pair, 'buy', TRADE_AMOUNT, entry_price)
                    await place_order(exchanges['kucoin'], pair, 'sell', TRADE_AMOUNT, take_profit_price)
        except Exception as e:
            logging.error(f"Ошибка в основном цикле арбитража: {e}")

        await asyncio.sleep(60)  # Задержка между циклами

# Запуск бота
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(arbitrage_bot())