import pandas as pd
import numpy as np
import os
import urllib.request
import json
from datetime import datetime, timedelta
import pytz
import yfinance as yf
import pandas_ta as ta

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
TIMEZONE_UY           = pytz.timezone('America/Montevideo')
CAPITAL               = 1000.0        # ← Actualizar antes de cada sesión
RIESGO_PCT            = 0.02          # 2% por trade
SL_MULTIPLICADOR      = 1.5           # 1.5×ATR — más allá de barridas
TP_MULTIPLICADOR      = 2.25          # R/R 1:1.5 natural

# ── Reglas de salida diaria (basadas en % del capital actual) ──────────────
STOP_PCT_DIARIO  = 0.10   # −10% del capital → CERRAR TODO
META_PCT_DIARIA  = 0.15   # +15% del capital → CERRAR TODO
ADX_LATERAL      = 20     # Si los 3 activos tienen ADX < este valor → mercado lateral

# Calculados automáticamente al iniciar
STOP_USD_DIARIO  = round(CAPITAL * STOP_PCT_DIARIO, 2)
META_USD_DIARIA  = round(CAPITAL * META_PCT_DIARIA, 2)

DRIVE_FOLDER   = r'C:\Users\Owner\Desktop\EG_trading'
ARCHIVO_SALIDA = 'dashboard_GASPA_v9.html'

ACTIVOS = {
    'BTC-USD': 'Bitcoin',
    'GC=F':    'Oro',
    'NQ=F':    'NASDAQ',
}

KEYWORDS_RELEVANTES = {
    "Oro":     ["gold", "inflation", "cpi", "pce", "gdp", "fed", "fomc", "powell",
                "interest rate", "nonfarm", "michigan", "consumer"],
    "Bitcoin": ["fed", "fomc", "interest rate", "inflation", "cpi", "gdp", "powell"],
    "NASDAQ":  ["gdp", "cpi", "pce", "fed", "fomc", "interest rate", "powell",
                "nonfarm", "unemployment", "ism", "retail", "housing", "consumer"],
}

MACRO_CONTEXT = {
    "Oro":     {"sesgo": "ALCISTA",   "color": "#7ec8a0",
                "factores": ["GDP débil → refugio", "PCE alto", "Tensión geopolítica"]},
    "Bitcoin": {"sesgo": "CAUTELOSO", "color": "#c8a87e",
                "factores": ["Correlación NASDAQ", "Resistencia 68k–72k"]},
    "NASDAQ":  {"sesgo": "CAUTELOSO", "color": "#c8a87e",
                "factores": ["GDP bajo expectativas", "PCE alto", "RSI elevado"]},
}

LOTES_ACTIVO = {
    'Bitcoin': {'completo': 0.03},
    'Oro':     {'completo': 0.01},
    'NASDAQ':  {'completo': 0.1},
}

# SL/TP en USD fijos de referencia (usado para display)
SL_USD_ACTIVO = {'Bitcoin': 20, 'Oro': 20, 'NASDAQ': 20}
TP_USD_ACTIVO = {'Bitcoin': 30, 'Oro': 30, 'NASDAQ': 30}


# ─── ANÁLISIS MULTI-TIMEFRAME ─────────────────────────────────────────────────
def analizar_activo(simbolo, nombre):
    resultado = {}
    for tf, period in [('1h', '30d'), ('15m', '10d'), ('5m', '5d')]:
        try:
            data = yf.download(simbolo, period=period, interval=tf,
                               progress=False, auto_adjust=True)
            if data is None or data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [str(c[0]) for c in data.columns]
            df = data.copy().bfill().ffill()

            df['RSI']    = ta.rsi(df['Close'], length=14)
            df['EMA_9']  = ta.ema(df['Close'], length=9)
            df['EMA_21'] = ta.ema(df['Close'], length=21)
            df['EMA_50'] = ta.ema(df['Close'], length=50)  # NUEVO: tendencia mayor

            macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                col = [c for c in macd_df.columns if 'MACD_' in c
                       and 'h' not in c.lower() and 's' not in c.lower()]
                sig_col = [c for c in macd_df.columns if 'MACDs_' in c]
                df['MACD'] = macd_df[col[0]] if col else 0
                df['MACD_SIGNAL'] = macd_df[sig_col[0]] if sig_col else 0
                df['MACD_HIST']   = df['MACD'] - df['MACD_SIGNAL']
            else:
                df['MACD'] = df['MACD_SIGNAL'] = df['MACD_HIST'] = 0

            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['ATR'] = atr if atr is not None else df['Close'] * 0.002

            adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            if adx_df is not None and not adx_df.empty:
                col_adx = [c for c in adx_df.columns if 'ADX_' in c and 'DM' not in c]
                col_dmp = [c for c in adx_df.columns if 'DMP_' in c]
                col_dmm = [c for c in adx_df.columns if 'DMN_' in c]
                df['ADX']  = adx_df[col_adx[0]] if col_adx else 25
                df['DI_P'] = adx_df[col_dmp[0]] if col_dmp else 20
                df['DI_M'] = adx_df[col_dmm[0]] if col_dmm else 20
            else:
                df['ADX'] = 25; df['DI_P'] = 20; df['DI_M'] = 20

            bb = ta.bbands(df['Close'], length=20, std=2)
            if bb is not None and not bb.empty:
                df['BB_upper'] = bb.iloc[:, 2]
                df['BB_lower'] = bb.iloc[:, 0]
                df['BB_mid']   = bb.iloc[:, 1]
                # Ancho de banda normalizado
                df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']
            else:
                df['BB_upper'] = df['Close'] * 1.02
                df['BB_lower'] = df['Close'] * 0.98
                df['BB_mid']   = df['Close']
                df['BB_width'] = 0.04

            # NUEVO: Stochastic RSI para scalping
            stoch = ta.stochrsi(df['Close'], length=14, rsi_length=14, k=3, d=3)
            if stoch is not None and not stoch.empty:
                k_col = [c for c in stoch.columns if 'STOCHRSIk' in c]
                d_col = [c for c in stoch.columns if 'STOCHRSId' in c]
                df['SRSI_K'] = stoch[k_col[0]] if k_col else 50
                df['SRSI_D'] = stoch[d_col[0]] if d_col else 50
            else:
                df['SRSI_K'] = df['SRSI_D'] = 50

            df = df.bfill().ffill()
            fila    = df.iloc[-1]
            fila_p  = df.iloc[-2]  # vela anterior

            precio  = float(fila['Close'])
            rsi     = round(float(fila['RSI']), 1)
            ema9    = float(fila['EMA_9'])
            ema21   = float(fila['EMA_21'])
            ema50   = float(fila['EMA_50'])
            macd    = float(fila['MACD'])
            macd_h  = float(fila['MACD_HIST'])
            macd_hp = float(fila_p['MACD_HIST'])
            atr_v   = round(float(fila['ATR']), 2)
            adx_v   = round(float(fila['ADX']), 1)
            di_p    = round(float(fila['DI_P']), 1)
            di_m    = round(float(fila['DI_M']), 1)
            bb_u    = float(fila['BB_upper'])
            bb_l    = float(fila['BB_lower'])
            bb_w    = round(float(fila['BB_width']) * 100, 2)  # en %
            srsi_k  = round(float(fila['SRSI_K']), 1)
            srsi_d  = round(float(fila['SRSI_D']), 1)

            puntos  = 0
            razones = []

            # RSI — sobreventa/sobrecompra con gradación
            if rsi < 25:    puntos += 25; razones.append('RSI extremo ↓')
            elif rsi < 35:  puntos += 15; razones.append('RSI sobreventa')
            elif rsi > 75:  puntos -= 25; razones.append('RSI extremo ↑')
            elif rsi > 65:  puntos -= 15; razones.append('RSI sobrecompra')
            elif 35 <= rsi <= 48: puntos += 10; razones.append('RSI zona compra')

            # EMA estructura
            if ema9 > ema21 > ema50:   puntos += 30; razones.append('EMAs alineadas ↑')
            elif ema9 > ema21:         puntos += 18; razones.append('EMA 9>21 ↑')
            elif ema9 < ema21 < ema50: puntos -= 30; razones.append('EMAs alineadas ↓')
            else:                      puntos -= 18; razones.append('EMA 9<21 ↓')

            # MACD — histograma con momentum (¿cruzando o divergiendo?)
            if macd > 0 and macd_h > 0 and macd_h > macd_hp:
                puntos += 20; razones.append('MACD momentum ↑')
            elif macd > 0:
                puntos += 10; razones.append('MACD+')
            elif macd < 0 and macd_h < 0 and macd_h < macd_hp:
                puntos -= 20; razones.append('MACD momentum ↓')
            else:
                puntos -= 10; razones.append('MACD−')

            # ADX + DI
            if adx_v > 30: puntos += 15; razones.append(f'Tendencia fuerte {adx_v:.0f}')
            elif adx_v > 20: puntos += 7; razones.append(f'ADX {adx_v:.0f}')
            # DI direccional
            if di_p > di_m: puntos += 8
            else: puntos -= 8

            # Bollinger
            if precio <= bb_l * 1.001:  puntos += 18; razones.append('BB inf')
            elif precio >= bb_u * 0.999: puntos -= 18; razones.append('BB sup')

            # Stochastic RSI — buen indicador para scalping
            if srsi_k < 20 and srsi_k > srsi_d:  puntos += 15; razones.append('StochRSI cruce ↑')
            elif srsi_k > 80 and srsi_k < srsi_d: puntos -= 15; razones.append('StochRSI cruce ↓')

            # Soporte/resistencia reciente (últimas 30 velas)
            ventana_sr = df.iloc[-30:]
            sr_resist  = float(ventana_sr['High'].max())
            sr_soporte = float(ventana_sr['Low'].min())
            rango_sr   = sr_resist - sr_soporte
            if rango_sr > 0:
                pos_precio = (precio - sr_soporte) / rango_sr
                if pos_precio <= 0.15:
                    puntos += 25; razones.append('En soporte')
                elif pos_precio <= 0.30:
                    puntos += 12; razones.append('Cerca soporte')
                elif pos_precio >= 0.85:
                    puntos -= 25; razones.append('En resistencia')
                elif pos_precio >= 0.70:
                    puntos -= 12; razones.append('Cerca resistencia')

            if puntos >= 50:    codigo = 'COMPRAR'
            elif puntos >= 25:  codigo = 'COMPRA_DEBIL'
            elif puntos <= -50: codigo = 'VENDER'
            elif puntos <= -25: codigo = 'VENTA_DEBIL'
            else:               codigo = 'NEUTRO'

            # SL ampliado 1.5×ATR para sobrevivir barridas
            sl_buy  = round(precio - SL_MULTIPLICADOR * atr_v, 2)
            tp_buy  = round(precio + TP_MULTIPLICADOR * atr_v, 2)
            sl_sell = round(precio + SL_MULTIPLICADOR * atr_v, 2)
            tp_sell = round(precio - TP_MULTIPLICADOR * atr_v, 2)

            resultado[tf] = {
                'precio':        precio,
                'rsi':           rsi,
                'adx':           adx_v,
                'di_p':          di_p,
                'di_m':          di_m,
                'atr':           atr_v,
                'ema_alcista':   ema9 > ema21,
                'ema50_alcista': precio > ema50,
                'macd_alcista':  macd > 0,
                'macd_hist':     round(macd_h, 4),
                'bb_width':      bb_w,
                'srsi_k':        srsi_k,
                'srsi_d':        srsi_d,
                'puntos':        puntos,
                'codigo':        codigo,
                'razon':         ' · '.join(razones[:3]),
                'sr_resist':     round(sr_resist, 2),
                'sr_soporte':    round(sr_soporte, 2),
                'sl_buy':  sl_buy,  'tp_buy':  tp_buy,
                'sl_sell': sl_sell, 'tp_sell': tp_sell,
                'bb_upper': round(bb_u, 2), 'bb_lower': round(bb_l, 2),
            }
        except Exception as e:
            print(f'   ⚠️  Error {nombre} {tf}: {e}')
    return resultado


# ─── DETECTOR STOP HUNT (mejorado) ───────────────────────────────────────────
def detectar_stop_hunt(simbolo):
    resultado = {
        'alerta':             False,
        'tipo':               None,
        'nivel_redondo':      False,
        'estructura_intacta': True,
        'resumen':            'Sin señal',
        'fuerza':             0,
    }
    try:
        d1m = yf.download(simbolo, period='1d', interval='1m',
                          progress=False, auto_adjust=True)
        if d1m is None or d1m.empty or len(d1m) < 25:
            return resultado
        if isinstance(d1m.columns, pd.MultiIndex):
            d1m.columns = [str(c[0]) for c in d1m.columns]
        d1m = d1m.bfill().ffill()

        # Analizar las últimas 3 velas, no solo la última
        alertas = []
        for i in [-1, -2, -3]:
            ult = d1m.iloc[i]
            precio_c = float(ult['Close'])
            cuerpo   = abs(float(ult['Close']) - float(ult['Open']))
            rango    = float(ult['High']) - float(ult['Low'])
            if rango == 0:
                continue
            mecha_inf = (float(ult['Open']) - float(ult['Low'])
                         if float(ult['Close']) > float(ult['Open'])
                         else float(ult['Close']) - float(ult['Low']))
            mecha_sup = (float(ult['High']) - float(ult['Close'])
                         if float(ult['Close']) > float(ult['Open'])
                         else float(ult['High']) - float(ult['Open']))

            vol_prom = float(d1m['Volume'].rolling(20).mean().iloc[i])
            vol_act  = float(d1m['Volume'].iloc[i])
            spike    = vol_act > 1.5 * vol_prom if vol_prom > 0 else False
            pos_c    = (float(ult['Close']) - float(ult['Low'])) / rango

            ml_inf = mecha_inf > 2 * cuerpo if cuerpo > 0 else False
            ml_sup = mecha_sup > 2 * cuerpo if cuerpo > 0 else False

            if ml_inf and spike and pos_c > 0.5:
                alertas.append(('ALCISTA', mecha_inf / rango * 100, vol_act / vol_prom if vol_prom > 0 else 1))
            elif ml_sup and spike and pos_c < 0.5:
                alertas.append(('BAJISTA', mecha_sup / rango * 100, vol_act / vol_prom if vol_prom > 0 else 1))

        # Estructura 15m
        d15m = yf.download(simbolo, period='5d', interval='15m',
                           progress=False, auto_adjust=True)
        estructura_intacta = True
        if d15m is not None and not d15m.empty and len(d15m) > 10:
            if isinstance(d15m.columns, pd.MultiIndex):
                d15m.columns = [str(c[0]) for c in d15m.columns]
            min_prev = float(d15m['Low'].iloc[-10:-1].min())
            cierre   = float(d15m['Close'].iloc[-1])
            if cierre < min_prev:
                estructura_intacta = False

        precio_actual = float(d1m['Close'].iloc[-1])
        redondos = [round(precio_actual / paso) * paso
                    for paso in [100, 500, 1000, 5000, 10000]]
        cerca_redondo = any(abs(precio_actual - r) / precio_actual < 0.002
                            for r in redondos if r > 0)

        if alertas and estructura_intacta:
            tipo_alerta = alertas[0][0]
            mecha_pct   = alertas[0][1]
            vol_ratio   = alertas[0][2]
            score       = len(alertas) + (1 if cerca_redondo else 0) + (1 if estructura_intacta else 0)
            dir_txt     = 'LONG' if tipo_alerta == 'ALCISTA' else 'SHORT'
            resultado.update({
                'alerta':             True,
                'tipo':               f'STOP HUNT {tipo_alerta}',
                'nivel_redondo':      cerca_redondo,
                'estructura_intacta': estructura_intacta,
                'resumen':            f'Barrida detectada — posible reingreso {dir_txt}',
                'fuerza':             score,
                'mecha_pct':          round(mecha_pct, 1),
                'vol_ratio':          round(vol_ratio, 2),
            })
        elif not estructura_intacta:
            resultado['resumen'] = 'Estructura 15m rota — puede ser reversión real'
        else:
            resultado['resumen'] = 'Movimiento normal'

    except Exception as e:
        print(f'   ⚠️  Error stop hunt {simbolo}: {e}')
    return resultado


# ─── FILTRO 1M ────────────────────────────────────────────────────────────────
def analizar_1m(simbolo):
    try:
        data = yf.download(simbolo, period='1d', interval='1m',
                           progress=False, auto_adjust=True)
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [str(c[0]) for c in data.columns]
        df = data.copy().bfill().ffill()
        df['RSI']        = ta.rsi(df['Close'], length=14)
        df['EMA_9']      = ta.ema(df['Close'], length=9)
        df['EMA_21']     = ta.ema(df['Close'], length=21)
        df['Volume_EMA'] = ta.ema(df['Volume'], length=10)
        df = df.dropna()
        if len(df) < 20:
            return None

        ultimas  = df.iloc[-10:]
        rsi_act  = round(float(df['RSI'].iloc[-1]), 1)

        mins_asc    = bool(ultimas['Low'].is_monotonic_increasing)
        div_alcista = bool(ultimas['Low'].iloc[-1] < ultimas['Low'].iloc[0]
                          and ultimas['RSI'].iloc[-1] > ultimas['RSI'].iloc[0])
        vol_osc     = bool(float(df['Volume'].iloc[-1]) > float(df['Volume_EMA'].iloc[-1]))
        ema_alc     = bool(float(df['EMA_9'].iloc[-1]) > float(df['EMA_21'].iloc[-1]))
        vol_crec    = bool(float(df['Volume'].iloc[-1]) > float(df['Volume'].iloc[-2]))

        conf_long  = sum([mins_asc, div_alcista, vol_osc, ema_alc, vol_crec])
        conf_short = sum([not mins_asc, not ema_alc, not vol_osc])

        return {
            'rsi_actual':          rsi_act,
            'minimos_ascendentes': mins_asc,
            'divergencia_alcista': div_alcista,
            'vol_osc_positivo':    vol_osc,
            'ema_alcista':         ema_alc,
            'vol_creciente':       vol_crec,
            'conf_long':           conf_long,
            'conf_short':          conf_short,
            'puede_long':          conf_long >= 3,   # Subido de 2→3, más exigente
            'puede_short':         conf_short >= 2,
        }
    except Exception as e:
        print(f'   ⚠️  Error filtro 1m: {e}')
        return None


# ─── SUGERENCIA DE TRADE ──────────────────────────────────────────────────────
def sugerir_trade(nombre, res_mtf, f1m, stop_hunt):
    try:
        dec, pts = decision_final(res_mtf)
        r_ref    = res_mtf.get('5m', res_mtf.get('15m'))
        if not r_ref:
            return None

        precio = r_ref['precio']
        atr    = r_ref['atr']
        lote   = LOTES_ACTIVO.get(nombre, {}).get('completo', 0.01)
        adx    = r_ref.get('adx', 0)
        bb_u   = r_ref.get('bb_upper', precio * 1.02)
        bb_l   = r_ref.get('bb_lower', precio * 0.98)

        if dec in ('COMPRAR', 'COMPRA_DEBIL'):
            direccion = 'LONG'
        elif dec in ('VENDER', 'VENTA_DEBIL'):
            direccion = 'SHORT'
        else:
            if f1m and f1m['puede_long']:    direccion = 'LONG'
            elif f1m and f1m['puede_short']: direccion = 'SHORT'
            else:                            direccion = 'LONG'

        # SL ampliado siempre (para sobrevivir barridas)
        mult_sl = SL_MULTIPLICADOR
        nota_sl = f'SL {mult_sl}×ATR — alejado de zona de barridas'
        if stop_hunt and stop_hunt.get('alerta'):
            mult_sl = SL_MULTIPLICADOR * 1.2
            nota_sl = f'⚡ SL {mult_sl:.1f}×ATR — stop hunt detectado, SL extra amplio'

        if direccion == 'LONG':
            sl_base = precio - mult_sl * atr
            tp_base = precio + TP_MULTIPLICADOR * atr
            # Alejar de nivel redondo
            for paso in [10000, 5000, 1000, 500, 100, 50, 10]:
                if precio > paso * 5:
                    nivel = round(sl_base / paso) * paso
                    if nivel > 0 and abs(sl_base - nivel) / precio < 0.001:
                        sl_base = nivel - paso * 0.15
                        nota_sl += ' · alejado de nivel redondo'
                        break
        else:
            sl_base = precio + mult_sl * atr
            tp_base = precio - TP_MULTIPLICADOR * atr
            for paso in [10000, 5000, 1000, 500, 100, 50, 10]:
                if precio > paso * 5:
                    nivel = round(sl_base / paso) * paso
                    if nivel > 0 and abs(sl_base - nivel) / precio < 0.001:
                        sl_base = nivel + paso * 0.15
                        nota_sl += ' · alejado de nivel redondo'
                        break

        sl = round(sl_base, 2)
        tp = round(tp_base, 2)

        dist_sl = abs(precio - sl)
        dist_tp = abs(precio - tp)
        rr      = round(dist_tp / dist_sl, 2) if dist_sl > 0 else 0

        # Calidad más exigente
        if dec in ('COMPRAR', 'VENDER') and f1m and ((direccion == 'LONG' and f1m['puede_long']) or (direccion == 'SHORT' and f1m['puede_short'])):
            calidad = 'ALTA'
        elif dec in ('COMPRAR', 'VENDER'):
            calidad = 'MEDIA'
        elif dec in ('COMPRA_DEBIL', 'VENTA_DEBIL') and f1m and f1m['puede_long']:
            calidad = 'MEDIA'
        else:
            calidad = 'BAJA'

        motivos_invalido = []
        if rr < 1.3:
            motivos_invalido.append(f'R/R {rr:.2f} < mín 1.3')
        if adx < 18:
            motivos_invalido.append(f'ADX {adx:.0f} — mercado lateral')
        if dec == 'NEUTRO' and (not f1m or (not f1m['puede_long'] and not f1m['puede_short'])):
            motivos_invalido.append('NEUTRO sin confirmación 1m')
        if calidad == 'BAJA':
            motivos_invalido.append('Señal débil')

        valido = len(motivos_invalido) == 0

        return {
            'valido':       valido,
            'motivo':       ' · '.join(motivos_invalido) if not valido else '',
            'direccion':    direccion,
            'dec_icm':      dec,
            'pts':          pts,
            'calidad':      calidad,
            'precio':       precio,
            'sl':           sl,
            'tp':           tp,
            'rr':           rr,
            'lote':         lote,
            'riesgo_usd':   round(CAPITAL * RIESGO_PCT, 2),
            'nota_sl':      nota_sl,
            'adx':          adx,
            'cerca_bb_inf': precio <= bb_l * 1.003,
            'cerca_bb_sup': precio >= bb_u * 0.997,
            'atr':          atr,
        }
    except Exception as e:
        print(f'   ⚠️  Error sugerencia trade {nombre}: {e}')
        return None


# ─── DXY ──────────────────────────────────────────────────────────────────────
def analizar_dxy():
    resultado = {}
    for tf, period in [('1h', '30d'), ('1d', '60d')]:
        try:
            data = yf.download('DX-Y.NYB', period=period, interval=tf,
                               progress=False, auto_adjust=True)
            if data is None or data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [str(c[0]) for c in data.columns]
            df = data.copy().bfill().ffill()
            df['RSI']    = ta.rsi(df['Close'], length=14)
            df['EMA_9']  = ta.ema(df['Close'], length=9)
            df['EMA_21'] = ta.ema(df['Close'], length=21)
            macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                col = [c for c in macd_df.columns if 'MACD_' in c
                       and 'h' not in c.lower() and 's' not in c.lower()]
                df['MACD'] = macd_df[col[0]] if col else 0
            else:
                df['MACD'] = 0
            adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
            if adx_df is not None and not adx_df.empty:
                col = [c for c in adx_df.columns if 'ADX_' in c and 'DM' not in c]
                df['ADX'] = adx_df[col[0]] if col else 25
            else:
                df['ADX'] = 25
            df   = df.dropna()
            fila = df.iloc[-1]

            rsi_v    = round(float(fila['RSI']), 1)
            adx_v    = round(float(fila['ADX']), 1)
            ema_alc  = float(fila['EMA_9']) > float(fila['EMA_21'])
            macd_alc = float(fila['MACD']) > 0

            if rsi_v > 55 and ema_alc:        fuerza = 'FUERTE'
            elif rsi_v < 45 and not ema_alc:  fuerza = 'DEBIL'
            else:                              fuerza = 'NEUTRAL'

            impactos = {
                'FUERTE':  'DXY fuerte — presión bajista en Oro y BTC',
                'DEBIL':   'DXY débil — favorable para Oro y BTC',
                'NEUTRAL': 'DXY sin dirección — seguí señales propias',
            }
            resultado[tf] = {
                'precio':       round(float(fila['Close']), 3),
                'rsi':          rsi_v,
                'adx':          adx_v,
                'ema_alcista':  ema_alc,
                'macd_alcista': macd_alc,
                'fuerza':       fuerza,
                'impacto':      impactos[fuerza],
            }
        except Exception as e:
            print(f'   ⚠️  Error DXY {tf}: {e}')
    return resultado


# ─── CALENDARIO ───────────────────────────────────────────────────────────────
def _detectar_activos(titulo):
    titulo_lower = titulo.lower()
    afectados = [a for a, kws in KEYWORDS_RELEVANTES.items()
                 if any(k in titulo_lower for k in kws)]
    return afectados if afectados else ['NASDAQ']

def _impacto_emoji(impact):
    return {'High': '🔴', 'Medium': '🟡', 'Low': '⚪'}.get(impact, '⚪')

def obtener_calendario():
    try:
        url = 'https://nfs.faireconomy.media/ff_calendar_thisweek.json'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = json.loads(r.read().decode('utf-8'))
    except Exception as e:
        print(f'   ⚠️  Calendario no disponible: {e}')
        return []

    calendario = []
    for ev in raw:
        if ev.get('country', '').upper() != 'USD': continue
        if ev.get('impact', 'Low') not in ('High', 'Medium'): continue
        fecha_raw = ev.get('date', '')
        try:
            dt_utc = datetime.fromisoformat(fecha_raw.replace('Z', '+00:00'))
            if dt_utc.tzinfo is None:
                dt_utc = pytz.utc.localize(dt_utc)
        except:
            continue
        dt_uy = dt_utc.astimezone(TIMEZONE_UY)
        dias_es = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        calendario.append({
            'dia':     f"{dias_es[dt_uy.weekday()]} {dt_uy.strftime('%d/%m')}",
            'hora_uy': dt_uy.strftime('%H:%M'),
            'evento':  ev.get('title', ''),
            'impacto': _impacto_emoji(ev.get('impact', 'Low')),
            'afecta':  _detectar_activos(ev.get('title', '')),
            '_dt_uy':  dt_uy,
        })
    calendario.sort(key=lambda x: x['_dt_uy'])
    print(f'   📅 {len(calendario)} eventos USD esta semana')
    return calendario

def hay_evento_rojo_proximo(calendario, minutos=30):
    ahora  = datetime.now(TIMEZONE_UY)
    limite = ahora + timedelta(minutes=minutos)
    for ev in calendario:
        if ev.get('impacto') == '🔴' and ahora <= ev.get('_dt_uy', ahora) <= limite:
            return ev
    return None


# ─── DECISIÓN FINAL ───────────────────────────────────────────────────────────
def decision_final(res):
    if not res:
        return 'NEUTRO', 0
    pesos = {'5m': 0.5, '15m': 0.3, '1h': 0.2}
    total = sum(res[tf]['puntos'] * pesos[tf] for tf in pesos if tf in res)
    pts   = round(total)
    if pts >= 50:    return 'COMPRAR', pts
    elif pts >= 25:  return 'COMPRA_DEBIL', pts
    elif pts <= -50: return 'VENDER', pts
    elif pts <= -25: return 'VENTA_DEBIL', pts
    else:            return 'NEUTRO', pts


# ─── BUILD HTML ───────────────────────────────────────────────────────────────
def build_mercado(hora, hora_str):

    datos = {}
    for simbolo, nombre in ACTIVOS.items():
        print(f'   📊 {nombre}...')
        res  = analizar_activo(simbolo, nombre)
        f1m  = analizar_1m(simbolo)
        if f1m:
            res['_1m'] = f1m
        sh   = detectar_stop_hunt(simbolo)
        res['_sh'] = sh
        trade = sugerir_trade(nombre, res, f1m, sh)
        res['_trade'] = trade
        datos[nombre] = res

    dxy = analizar_dxy()

    # Correlaciones
    precios = {}
    for simbolo, nombre in ACTIVOS.items():
        try:
            data = yf.download(simbolo, period='10d', interval='1h', progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                precios[nombre] = data['Close'].reset_index(drop=True)
        except: pass

    correlaciones = {}
    nombres_list  = list(precios.keys())
    for i in range(len(nombres_list)):
        for j in range(i + 1, len(nombres_list)):
            a, b = nombres_list[i], nombres_list[j]
            try:
                ml = min(len(precios[a]), len(precios[b]))
                if ml > 10:
                    correlaciones[f'{a}|{b}'] = round(
                        float(precios[a].iloc[-ml:].corr(precios[b].iloc[-ml:])), 2)
            except:
                correlaciones[f'{a}|{b}'] = 0.0

    CALENDARIO  = obtener_calendario()
    evento_rojo = hay_evento_rojo_proximo(CALENDARIO, minutos=30)
    dia_actual  = hora.strftime('%d/%m')
    sesion_ok   = 13 <= hora.hour <= 17

    # ── Paleta de colores
    C = {
        'bg':       '#0f1117',
        'bg2':      '#161b24',
        'bg3':      '#1c2333',
        'border':   '#252d3d',
        'border2':  '#2a3447',
        'text':     '#c8d8e8',
        'text2':    '#7a8fa8',
        'text3':    '#4a5f78',
        'buy':      '#5eb87a',    # verde suave
        'buy_bg':   '#1a2e22',
        'sell':     '#c87878',    # rojo suave
        'sell_bg':  '#2e1a1a',
        'warn':     '#c8a84e',    # ámbar
        'warn_bg':  '#2e2414',
        'blue':     '#6aaecc',
        'purple':   '#9a8acc',
        'neutral':  '#4a5f78',
    }

    # ── Detectar si todos los activos están laterales
    adx_valores = []
    for nombre_a, res_a in datos.items():
        r_ref_a = res_a.get('5m', res_a.get('15m', {}))
        if r_ref_a:
            adx_valores.append(r_ref_a.get('adx', 99))
    mercado_lateral = len(adx_valores) == len(ACTIVOS) and all(v < ADX_LATERAL for v in adx_valores)

    stop_usd = round(CAPITAL * STOP_PCT_DIARIO, 2)
    meta_usd = round(CAPITAL * META_PCT_DIARIA, 2)

    lateral_banner = ''
    if mercado_lateral:
        lateral_banner = f"""
    <div style="background:#2e2414;border:1px solid #c8a84e;border-radius:8px;padding:12px 18px;
                margin:0 20px 4px 20px;display:flex;align-items:center;gap:10px;font-size:0.82em">
        <span style="font-size:1.3em">&#x1F7E1;</span>
        <div>
            <div style="color:#c8a84e;font-weight:700">MERCADO LATERAL — ADX &lt; {ADX_LATERAL} en todos los activos</div>
            <div style="color:#7a6a4a;margin-top:2px">No hay tendencia clara. Lo mejor que podes hacer hoy es <strong style="color:#c8a84e">no operar</strong>.</div>
        </div>
    </div>"""

    # ────────────────────────────────────────────────────────────
    # REGLAS PSICOLÓGICAS — bloque fijo visible siempre arriba
    # ────────────────────────────────────────────────────────────
    psico_html = f"""
    {lateral_banner}
    <div class="psico-rules">
        <div class="psico-title">PROTOCOLO DE TRADING &nbsp;·&nbsp; Capital actual ${CAPITAL:,.2f} &nbsp;·&nbsp; Stop dia −${stop_usd} &nbsp;·&nbsp; Meta dia +${meta_usd}</div>
        <div class="psico-grid">
            <div class="psico-rule critical">
                <div class="rule-num">01</div>
                <div class="rule-text">Perdiste −{STOP_PCT_DIARIO*100:.0f}% hoy (−${stop_usd})<br><strong>CERRAR PLATAFORMA — ya</strong></div>
            </div>
            <div class="psico-rule critical">
                <div class="rule-num">02</div>
                <div class="rule-text">Ganaste +{META_PCT_DIARIA*100:.0f}% hoy (+${meta_usd})<br><strong>CERRAR Y CELEBRAR — ya</strong></div>
            </div>
            <div class="psico-rule critical">
                <div class="rule-num">03</div>
                <div class="rule-text">ADX &lt; {ADX_LATERAL} en todos los activos<br><strong>MERCADO LATERAL — no operar</strong></div>
            </div>
            <div class="psico-rule">
                <div class="rule-num">04</div>
                <div class="rule-text">El lote es siempre fijo<br><strong>NO agranda cuando perdes</strong></div>
            </div>
            <div class="psico-rule">
                <div class="rule-num">05</div>
                <div class="rule-text">Despues de 2 perdidas seguidas<br><strong>PAUSA 15 min — obligatoria</strong></div>
            </div>
            <div class="psico-rule accent">
                <div class="rule-num">06</div>
                <div class="rule-text">Si el mercado barrio tu SL<br><strong>espera confirmacion — no reentrar a ciegas</strong></div>
            </div>
        </div>
    </div>"""

    # ────────────────────────────────────────────────────────────
    # CARDS por activo
    # ────────────────────────────────────────────────────────────
    ICONS = {'Bitcoin': '₿', 'Oro': 'Au', 'NASDAQ': 'NQ'}
    DEC_STYLES = {
        'COMPRAR':      (C['buy'],    C['buy_bg'],  '▲ COMPRAR'),
        'COMPRA_DEBIL': (C['warn'],   C['warn_bg'], '▲ DÉBIL'),
        'VENDER':       (C['sell'],   C['sell_bg'], '▼ VENDER'),
        'VENTA_DEBIL':  (C['sell'],   C['sell_bg'], '▼ DÉBIL'),
        'NEUTRO':       (C['neutral'],C['bg3'],     '— ESPERAR'),
    }

    cards_html = ''
    for nombre, res in datos.items():
        if not res:
            continue
        dec, pts  = decision_final(res)
        dc        = dec
        clr, cbg, clbl = DEC_STYLES.get(dc, DEC_STYLES['NEUTRO'])
        precio_ref = res.get('5m', res.get('15m', res.get('1h', {}))).get('precio', 0)
        icon       = ICONS.get(nombre, '●')
        macro      = MACRO_CONTEXT.get(nombre, {})
        sesgo      = macro.get('sesgo', '---')
        sesgo_clr  = macro.get('color', C['text2'])
        factores   = macro.get('factores', [])
        trade      = res.get('_trade')
        sh         = res.get('_sh', {})
        f1m        = res.get('_1m')

        # Porcentaje score
        spct = min(max((pts + 70) / 140 * 100, 0), 100)

        # ── Timeframe rows
        tf_rows = ''
        for tf in ['1h', '15m', '5m']:
            if tf not in res:
                continue
            r    = res[tf]
            tc, tbg, tlbl = DEC_STYLES.get(r['codigo'], DEC_STYLES['NEUTRO'])
            peso = {'5m': '50%', '15m': '30%', '1h': '20%'}.get(tf, '')
            rsi_c = (C['sell'] if r['rsi'] > 68 else C['buy'] if r['rsi'] < 33 else
                     C['warn'] if r['rsi'] < 45 else C['text2'])
            adx_c = C['buy'] if r['adx'] > 25 else C['text3']
            srsi_k = r.get('srsi_k', 50)
            srsi_c = C['buy'] if srsi_k < 25 else C['sell'] if srsi_k > 75 else C['text2']

            tf_rows += f"""
            <div class="tf-row">
                <div class="tf-label">
                    <span class="tf-name">{tf}</span>
                    <span class="tf-weight">{peso}</span>
                </div>
                <div class="tf-signal" style="color:{tc};background:{tbg};border-color:{tc}40">{tlbl}</div>
                <div class="tf-pts" style="color:{tc}">{r['puntos']:+d}</div>
                <div class="tf-rsi" style="color:{rsi_c}">{r['rsi']}</div>
                <div class="tf-adx" style="color:{adx_c}">{r['adx']}</div>
                <div class="tf-srsi" style="color:{srsi_c}">{srsi_k}</div>
                <div class="tf-sr">
                    <span style="color:{C['buy']}88">S {r['sr_soporte']:,.0f}</span>
                    <span style="color:{C['sell']}88">R {r['sr_resist']:,.0f}</span>
                </div>
            </div>"""

        # ── Filtro 1m
        blk_1m = ''
        if f1m:
            lc  = C['buy'] if f1m['puede_long'] else C['text3']
            sc  = C['buy'] if f1m['puede_short'] else C['text3']
            blk_1m = f"""
            <div class="blk-1m">
                <div class="blk-title">FILTRO 1M &nbsp;<span style="color:{C['warn'] if f1m['rsi_actual']>65 else C['buy'] if f1m['rsi_actual']<35 else C['text2']}">{f1m['rsi_actual']}</span></div>
                <div class="chips-row">
                    {''.join([f'<span class="chip {"on" if v else "off"}">{t}</span>' for v, t in [
                        (f1m['minimos_ascendentes'], 'Mín↑'),
                        (f1m['divergencia_alcista'], 'Div RSI'),
                        (f1m['vol_osc_positivo'], 'Vol+'),
                        (f1m['ema_alcista'], 'EMA↑'),
                        (f1m['vol_creciente'], 'VolCr'),
                    ]])}
                </div>
                <div class="dir-row">
                    <div class="dir-btn" style="color:{lc};border-color:{lc}40;background:{lc}0d">▲ LONG {f1m['conf_long']}/5</div>
                    <div class="dir-btn" style="color:{sc};border-color:{sc}40;background:{sc}0d">▼ SHORT {f1m['conf_short']}/3</div>
                </div>
            </div>"""

        # ── Stop hunt
        blk_sh = ''
        if sh:
            if sh.get('alerta'):
                sh_c   = C['warn']
                sh_ico = '⚡'
                sh_txt = sh.get('resumen', '')
                sh_det = f'Mecha {sh.get("mecha_pct",0):.0f}% · Vol {sh.get("vol_ratio",0):.1f}× · ' + ('Nivel redondo' if sh.get('nivel_redondo') else '')
            elif not sh.get('estructura_intacta', True):
                sh_c   = C['sell']
                sh_ico = '⚠'
                sh_txt = sh.get('resumen', '')
                sh_det = 'Estructura 15m rota — podría ser reversión real'
            else:
                sh_c   = C['buy']
                sh_ico = '·'
                sh_txt = sh.get('resumen', 'Movimiento normal')
                sh_det = ''
            blk_sh = f"""
            <div class="blk-sh" style="border-color:{sh_c}30">
                <span class="sh-ico" style="color:{sh_c}">{sh_ico}</span>
                <div>
                    <div class="sh-txt" style="color:{sh_c}">{sh_txt}</div>
                    {f'<div class="sh-det">{sh_det}</div>' if sh_det else ''}
                </div>
            </div>"""

        # ── Trade suggestion
        blk_trade = ''
        if trade:
            t_es_long  = trade['direccion'] == 'LONG'
            t_clr      = C['buy'] if t_es_long else C['sell']
            t_bg       = C['buy_bg'] if t_es_long else C['sell_bg']
            t_dir      = '▲ LONG' if t_es_long else '▼ SHORT'
            rr         = trade['rr']
            rr_c       = C['buy'] if rr >= 1.5 else C['warn'] if rr >= 1.3 else C['sell']
            cal_map    = {'ALTA': (C['buy'], '●●●'), 'MEDIA': (C['warn'], '●●○'), 'BAJA': (C['text3'], '●○○')}
            cal_c, cal_s = cal_map.get(trade['calidad'], (C['text3'], '○○○'))

            if trade['valido']:
                header_txt  = f'{t_dir} — ENTRAR'
                header_clr  = t_clr
            else:
                header_txt  = f'NO OPERAR — {trade["motivo"]}'
                header_clr  = C['text3']
                t_bg        = C['bg3']

            # Barra visual SL→PRECIO→TP
            dist_sl = abs(trade['precio'] - trade['sl'])
            dist_tp = abs(trade['precio'] - trade['tp'])
            tot     = dist_sl + dist_tp
            sl_pct  = round(dist_sl / tot * 100) if tot > 0 else 50
            tp_pct  = 100 - sl_pct

            blk_trade = f"""
            <div class="blk-trade" style="background:{t_bg}80;border-color:{header_clr if trade['valido'] else C['border']}">
                <div class="trade-header">
                    <div>
                        <div class="trade-label">SUGERENCIA</div>
                        <div class="trade-dir" style="color:{header_clr}">{header_txt}</div>
                        <div class="trade-meta">{trade['dec_icm']} · {trade['pts']:+d}pts</div>
                    </div>
                    <div class="trade-quality" style="color:{cal_c}">{cal_s}<br><span style="font-size:0.7em">{trade['calidad']}</span></div>
                </div>
                <div class="trade-levels">
                    <div class="lvl sl">
                        <div class="lvl-label">SL</div>
                        <div class="lvl-price" style="color:{C['sell']}">{trade['sl']:,.2f}</div>
                        <div class="lvl-dist">−{dist_sl:,.1f}</div>
                    </div>
                    <div class="lvl entry">
                        <div class="lvl-label">ENTRADA</div>
                        <div class="lvl-price" style="color:{C['blue']}">{trade['precio']:,.2f}</div>
                        <div class="lvl-dist">precio actual</div>
                    </div>
                    <div class="lvl tp">
                        <div class="lvl-label">TP</div>
                        <div class="lvl-price" style="color:{C['buy']}">{trade['tp']:,.2f}</div>
                        <div class="lvl-dist">+{dist_tp:,.1f}</div>
                    </div>
                </div>
                <div class="trade-bar">
                    <div style="width:{sl_pct}%;background:{C['sell']};opacity:0.6;border-radius:3px 0 0 3px"></div>
                    <div style="width:3px;background:{C['blue']};flex-shrink:0"></div>
                    <div style="width:{tp_pct}%;background:{C['buy']};opacity:0.6;border-radius:0 3px 3px 0"></div>
                </div>
                <div class="trade-meta-row">
                    <span style="color:{rr_c}">R/R 1:{rr:.2f}</span>
                    <span>Lote {trade['lote']}</span>
                    <span style="color:{C['sell']}">Riesgo ${trade['riesgo_usd']}</span>
                </div>
                <div class="trade-sl-note">{trade['nota_sl']}</div>
            </div>"""

        cards_html += f"""
        <div class="asset-card" style="border-color:{clr}25;border-left-color:{clr}">
            <!-- HEADER -->
            <div class="card-header" style="border-color:{C['border']}">
                <div class="card-id">
                    <div class="card-icon" style="color:{clr};background:{cbg};border-color:{clr}30">{icon}</div>
                    <div>
                        <div class="card-name">{nombre}</div>
                        <div class="card-price">{precio_ref:,.2f}</div>
                    </div>
                    <div class="card-macro" style="color:{sesgo_clr};border-color:{sesgo_clr}30;background:{sesgo_clr}10">{sesgo}</div>
                </div>
                <div class="card-decision">
                    <div class="dec-badge" style="color:{clr};background:{cbg};border-color:{clr}40">{clbl}</div>
                    <div class="dec-score">
                        <span style="color:{C['text2']}">score</span>
                        <span style="color:{clr};font-weight:700">{pts:+d}</span>
                        <div class="score-bar"><div style="width:{spct:.0f}%;background:{clr}"></div></div>
                    </div>
                </div>
            </div>

            <!-- CUERPO 2 COLUMNAS -->
            <div class="card-body">
                <!-- COL IZQ: timeframes -->
                <div class="col-left">
                    <div class="tf-header">
                        <span>TF</span><span>SEÑAL</span><span>PTS</span>
                        <span>RSI</span><span>ADX</span><span>StRSI</span><span>S/R</span>
                    </div>
                    {tf_rows}
                    <div class="factors-row">
                        {''.join([f'<span class="factor-tag">{f}</span>' for f in factores])}
                    </div>
                </div>

                <!-- COL DER: filtros + trade -->
                <div class="col-right">
                    {blk_1m}
                    {blk_sh}
                    {blk_trade}
                </div>
            </div>
        </div>"""

    # ── DXY
    ref   = dxy.get('1h', dxy.get('1d', {})) if dxy else {}
    dc    = ref.get('fuerza', 'NEUTRAL')
    dc_c  = C['sell'] if dc == 'FUERTE' else C['buy'] if dc == 'DEBIL' else C['neutral']
    dxy_html = f"""
    <div class="section-block">
        <div class="section-title">DÓLAR · DXY</div>
        <div class="dxy-row">
            <div>
                <div class="dxy-price">{ref.get('precio', '—')}</div>
                <div class="dxy-label" style="color:{dc_c}">{dc}</div>
            </div>
            <div class="dxy-chips">
                <span style="color:{C['warn'] if ref.get('rsi',50)>55 else C['buy'] if ref.get('rsi',50)<45 else C['text2']}">RSI {ref.get('rsi','—')}</span>
                <span style="color:{C['buy'] if ref.get('adx',0)>25 else C['text3']}">ADX {ref.get('adx','—')}</span>
                <span style="color:{C['buy'] if ref.get('ema_alcista') else C['sell']}">EMA {'↑' if ref.get('ema_alcista') else '↓'}</span>
                <span style="color:{C['buy'] if ref.get('macd_alcista') else C['sell']}">MACD {'↑' if ref.get('macd_alcista') else '↓'}</span>
            </div>
            <div class="dxy-impacto" style="color:{dc_c}">{ref.get('impacto','—')}</div>
        </div>
    </div>"""

    # ── Calendario
    alerta_html = ''
    if evento_rojo:
        alerta_html = f"""
        <div class="event-alert">
            🚨 <strong>EVENTO ROJO EN &lt;30 MIN — NO ENTRAR</strong>
            <div>{evento_rojo['evento']} · {evento_rojo['hora_uy']} UY</div>
        </div>"""

    cal_rows = alerta_html
    if not CALENDARIO:
        cal_rows += f'<div style="color:{C["text3"]};padding:12px;font-size:0.8em">Sin datos de calendario</div>'
    else:
        for ev in CALENDARIO:
            ya_paso = '_dt_uy' in ev and ev['_dt_uy'] < datetime.now(TIMEZONE_UY)
            es_hoy  = dia_actual in ev['dia']
            op      = 'opacity:0.3;' if ya_paso else ''
            bg      = f'background:{C["bg3"]};' if es_hoy and not ya_paso else ''
            tags    = ''.join([f'<span class="ev-tag">{a}</span>' for a in ev['afecta']])
            cal_rows += f'<div class="cal-row" style="{op}{bg}"><span class="cal-dia">{ev["dia"]}</span><span class="cal-hora">{ev["hora_uy"]}</span><span>{ev["impacto"]}</span><span class="cal-ev">{ev["evento"]}{"  ✓" if ya_paso else ""}</span><span class="cal-tags">{tags}</span></div>'

    cal_html = f"""
    <div class="section-block">
        <div class="section-title">CALENDARIO ECONÓMICO</div>
        {cal_rows}
    </div>"""

    # ── Correlaciones
    corr_rows = ''
    for par, valor in correlaciones.items():
        a, b = par.split('|')
        c_c  = C['buy'] if valor >= 0.6 else C['sell'] if valor <= -0.6 else C['neutral']
        c_lbl = 'Juntos' if valor >= 0.6 else 'Opuestos' if valor <= -0.6 else 'Independ.'
        pct  = int(abs(valor) * 100)
        corr_rows += f"""
        <div class="corr-row">
            <div class="corr-pair"><span style="color:{C['blue']}">{a}</span> · <span style="color:{C['warn']}">{b}</span></div>
            <div class="corr-bar"><div style="width:{pct}%;background:{c_c};opacity:0.7"></div></div>
            <div class="corr-val" style="color:{c_c}">{valor:+.2f}</div>
            <div class="corr-lbl" style="color:{c_c}">{c_lbl}</div>
        </div>"""

    corr_html = f"""
    <div class="section-block">
        <div class="section-title">CORRELACIONES · 10 jornadas 1h</div>
        {corr_rows}
    </div>"""

    # ── Banner sesión
    sesion_html = f'<span class="sesion-badge {"ok" if sesion_ok else "off"}">{"✓ Sesión activa" if sesion_ok else "⏳ Fuera de sesión"}</span>'

    return psico_html, cards_html, dxy_html, cal_html, corr_html, sesion_html


# ─── CSS ──────────────────────────────────────────────────────────────────────
def get_css():
    return """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0f1117;
    --bg2:      #161b24;
    --bg3:      #1c2333;
    --border:   #252d3d;
    --border2:  #2a3447;
    --text:     #c8d8e8;
    --text2:    #7a8fa8;
    --text3:    #4a5f78;
    --buy:      #5eb87a;
    --buy-bg:   #1a2e22;
    --sell:     #c87878;
    --sell-bg:  #2e1a1a;
    --warn:     #c8a84e;
    --warn-bg:  #2e2414;
    --blue:     #6aaecc;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'IBM Plex Sans', sans-serif;
}

* { margin:0; padding:0; box-sizing:border-box; }

body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.5;
    min-height: 100vh;
}

/* ── HEADER ─────────────────────────────────────────── */
.top-header {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}
.header-left .sys-name {
    font-family: var(--mono);
    font-size: 0.65em;
    letter-spacing: 4px;
    color: var(--text3);
    text-transform: uppercase;
    margin-bottom: 4px;
}
.header-left .title {
    font-size: 1.3em;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.3px;
}
.header-right {
    font-family: var(--mono);
    text-align: right;
}
.header-right .hora {
    font-size: 0.85em;
    color: var(--blue);
    font-weight: 600;
}
.header-right #countdown {
    font-size: 0.7em;
    color: var(--text3);
    margin-top: 3px;
}
.sesion-badge {
    font-size: 0.75em;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    margin-top: 4px;
    display: inline-block;
}
.sesion-badge.ok  { background: #1a2e22; color: var(--buy); border: 1px solid #2e4a32; }
.sesion-badge.off { background: var(--warn-bg); color: var(--warn); border: 1px solid #4a3820; }

/* ── BARRA PARÁMETROS ────────────────────────────────── */
.params-bar {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 8px 24px;
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    font-family: var(--mono);
    font-size: 0.72em;
    color: var(--text3);
}
.params-bar .p-item { display: flex; gap: 5px; align-items: center; }
.params-bar .p-val  { color: var(--text2); font-weight: 600; }
.params-bar .p-stop { color: var(--sell); font-weight: 700; }
.params-bar .p-go   { color: var(--buy);  font-weight: 700; }

/* ── PROTOCOLO PSICOLÓGICO ───────────────────────────── */
.psico-rules {
    margin: 16px 20px 8px 20px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
}
.psico-title {
    font-family: var(--mono);
    font-size: 0.62em;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 12px;
}
.psico-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}
@media (max-width:800px) { .psico-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width:500px) { .psico-grid { grid-template-columns: 1fr; } }

.psico-rule {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 10px 12px;
}
.psico-rule.critical { border-color: #c87878;  background: #2e1a1a80; }
.psico-rule.accent   { border-color: #6aaecc30; background: #0d1a2280; }
.rule-num {
    font-family: var(--mono);
    font-size: 1.1em;
    font-weight: 700;
    color: var(--text3);
    flex-shrink: 0;
    padding-top: 1px;
}
.psico-rule.critical .rule-num { color: var(--sell); }
.psico-rule.accent   .rule-num { color: var(--blue); }
.rule-text {
    font-size: 0.78em;
    color: var(--text2);
    line-height: 1.4;
}
.rule-text strong { color: var(--text); font-weight: 600; }
.psico-rule.critical .rule-text strong { color: var(--sell); }
.psico-rule.accent   .rule-text strong { color: var(--blue); }

/* ── ASSET CARDS ─────────────────────────────────────── */
.cards-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 10px 20px;
}
.asset-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 3px solid transparent;
    border-radius: 10px;
    overflow: hidden;
}

/* Header de card */
.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg3);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
    gap: 10px;
}
.card-id {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
}
.card-icon {
    width: 40px; height: 40px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono);
    font-size: 0.95em;
    font-weight: 700;
    border: 1px solid transparent;
    flex-shrink: 0;
}
.card-name  { font-size: 1.2em; font-weight: 600; letter-spacing: -0.2px; }
.card-price { font-family: var(--mono); font-size: 1.0em; color: var(--blue); font-weight: 600; margin-top: 2px; }
.card-macro {
    font-size: 0.7em; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1px; padding: 3px 10px; border-radius: 20px;
    border: 1px solid transparent;
}
.card-decision { display: flex; flex-direction: column; align-items: flex-end; gap: 5px; }
.dec-badge {
    font-size: 0.9em; font-weight: 700; padding: 6px 14px;
    border-radius: 6px; border: 1px solid transparent;
    letter-spacing: 0.3px;
}
.dec-score {
    display: flex; align-items: center; gap: 7px;
    font-family: var(--mono); font-size: 0.8em;
}
.score-bar {
    width: 70px; height: 5px;
    background: var(--bg); border-radius: 3px; overflow: hidden;
}
.score-bar div { height: 100%; border-radius: 3px; transition: width 0.3s; }

/* Cuerpo 2 col */
.card-body {
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
}
@media (max-width:900px) { .card-body { grid-template-columns: 1fr; } }

.col-left  { border-right: 1px solid var(--border); padding: 0 14px; }
.col-right { padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; }

/* TF rows */
.tf-header {
    display: grid;
    grid-template-columns: 44px 80px 44px 44px 44px 52px 1fr;
    gap: 4px;
    padding: 6px 0 5px 0;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.62em;
    text-transform: uppercase;
    color: var(--text3);
    font-weight: 700;
}
.tf-row {
    display: grid;
    grid-template-columns: 44px 80px 44px 44px 44px 52px 1fr;
    gap: 4px;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    align-items: center;
    font-family: var(--mono);
    font-size: 0.82em;
}
.tf-label { display: flex; flex-direction: column; }
.tf-name  { font-weight: 700; color: var(--text2); font-size: 0.95em; }
.tf-weight{ font-size: 0.65em; color: var(--text3); }
.tf-signal{
    font-size: 0.75em; font-weight: 800; padding: 3px 6px;
    border-radius: 4px; border: 1px solid transparent;
    text-align: center;
}
.tf-pts, .tf-rsi, .tf-adx, .tf-srsi { text-align: center; font-weight: 700; }
.tf-sr  {
    display: flex; flex-direction: column;
    font-size: 0.7em; gap: 1px;
}
.factors-row {
    padding: 8px 0 6px 0;
    display: flex; flex-wrap: wrap; gap: 4px;
}
.factor-tag {
    font-size: 0.68em; color: var(--text3);
    background: var(--bg3); border: 1px solid var(--border);
    padding: 2px 7px; border-radius: 3px;
}

/* Bloques derecha */
.blk-1m {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 10px 12px;
}
.blk-title {
    font-family: var(--mono);
    font-size: 0.7em; text-transform: uppercase; letter-spacing: 1px;
    color: var(--text3); font-weight: 700; margin-bottom: 7px;
}
.chips-row  { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 7px; }
.chip {
    font-size: 0.72em; font-weight: 700; padding: 2px 8px;
    border-radius: 20px; border: 1px solid;
}
.chip.on  { color: var(--buy);  background: #1a2e2280; border-color: #3a6a4480; }
.chip.off { color: var(--text3); background: transparent; border-color: var(--border); }
.dir-row { display: grid; grid-template-columns: 1fr 1fr; gap: 5px; }
.dir-btn {
    font-size: 0.75em; font-weight: 700; text-align: center;
    padding: 6px 4px; border-radius: 5px; border: 1px solid transparent;
}

.blk-sh {
    display: flex; gap: 8px; align-items: flex-start;
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 7px; padding: 9px 11px;
}
.sh-ico  { font-size: 1.1em; flex-shrink: 0; margin-top: 1px; }
.sh-txt  { font-size: 0.78em; font-weight: 600; line-height: 1.3; }
.sh-det  { font-size: 0.68em; color: var(--text3); margin-top: 3px; font-family: var(--mono); }

/* Trade block */
.blk-trade {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 11px 13px;
}
.trade-header {
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 10px;
}
.trade-label {
    font-family: var(--mono); font-size: 0.6em; text-transform: uppercase;
    letter-spacing: 2px; color: var(--text3); margin-bottom: 3px;
}
.trade-dir   { font-size: 1.0em; font-weight: 700; }
.trade-meta  { font-size: 0.72em; color: var(--text3); margin-top: 3px; font-family: var(--mono); }
.trade-quality { text-align: right; font-family: var(--mono); font-size: 0.85em; font-weight: 700; letter-spacing: 2px; }
.trade-levels {
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px;
    margin-bottom: 8px;
}
.lvl { text-align: center; }
.lvl-label { font-size: 0.62em; text-transform: uppercase; color: var(--text3); letter-spacing: 1px; font-weight: 700; margin-bottom: 3px; }
.lvl-price { font-family: var(--mono); font-size: 0.88em; font-weight: 800; }
.lvl-dist  { font-size: 0.65em; color: var(--text3); font-family: var(--mono); margin-top: 2px; }
.trade-bar {
    display: flex; height: 5px; border-radius: 3px;
    overflow: hidden; margin-bottom: 7px; gap: 1px;
}
.trade-meta-row {
    display: flex; gap: 12px; font-family: var(--mono); font-size: 0.72em;
    margin-bottom: 5px;
}
.trade-sl-note {
    font-family: var(--mono); font-size: 0.62em; color: var(--text3);
    background: var(--bg); border-radius: 3px; padding: 3px 7px;
}

/* ── SECCIONES INFERIORES ────────────────────────────── */
.bottom-sections {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    padding: 10px 20px 20px 20px;
}
@media (max-width:1000px) { .bottom-sections { grid-template-columns: 1fr; } }

.section-block {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
}
.section-title {
    font-family: var(--mono);
    font-size: 0.62em; letter-spacing: 3px;
    text-transform: uppercase; color: var(--text3);
    font-weight: 700; margin-bottom: 12px;
}

/* DXY */
.dxy-row   { display: flex; flex-direction: column; gap: 8px; }
.dxy-price { font-family: var(--mono); font-size: 1.4em; font-weight: 700; color: var(--blue); }
.dxy-label { font-size: 0.75em; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.dxy-chips { display: flex; flex-wrap: wrap; gap: 8px; font-family: var(--mono); font-size: 0.78em; }
.dxy-impacto { font-size: 0.78em; color: var(--text2); padding: 6px 9px; background: var(--bg3); border-radius: 5px; }

/* Calendario */
.event-alert {
    background: #2e1a1a; border: 1px solid #c87878;
    border-radius: 6px; padding: 10px 13px; margin-bottom: 8px;
    font-size: 0.8em; color: var(--sell); line-height: 1.5;
}
.cal-row {
    display: grid;
    grid-template-columns: 70px 50px 20px 1fr auto;
    gap: 6px; align-items: center;
    padding: 6px 9px; border-radius: 5px;
    margin-bottom: 3px; font-size: 0.72em;
    border: 1px solid var(--border);
}
.cal-dia  { color: var(--text3); font-size: 0.88em; }
.cal-hora { font-family: var(--mono); color: var(--blue); font-size: 0.85em; }
.cal-ev   { color: var(--text); }
.cal-tags { display: flex; flex-wrap: wrap; gap: 3px; }
.ev-tag   {
    font-size: 0.75em; background: var(--bg3); color: var(--text3);
    padding: 1px 5px; border-radius: 3px; border: 1px solid var(--border);
}

/* Correlaciones */
.corr-row {
    display: grid;
    grid-template-columns: 130px 1fr 40px 80px;
    gap: 8px; align-items: center;
    padding: 7px 0; border-bottom: 1px solid var(--border);
    font-size: 0.75em;
}
.corr-pair { font-family: var(--mono); font-weight: 600; }
.corr-bar  { height: 4px; background: var(--bg3); border-radius: 2px; overflow: hidden; }
.corr-bar div { height: 100%; border-radius: 2px; }
.corr-val  { font-family: var(--mono); font-weight: 700; text-align: center; }
.corr-lbl  { font-size: 0.85em; font-weight: 600; }

/* ── FOOTER ──────────────────────────────────────────── */
.footer {
    text-align: center; padding: 14px;
    color: var(--text3); font-size: 0.68em;
    font-family: var(--mono); border-top: 1px solid var(--border);
}
"""


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def generar():
    hora     = datetime.now(TIMEZONE_UY)
    hora_str = hora.strftime('%d/%m/%Y %H:%M')

    print('=' * 65)
    print('📊 Sistema GASPA v9.0 — Dashboard Mercado')
    print('=' * 65)
    print(f'⏰ {hora_str} UY\n')

    psico_html, cards_html, dxy_html, cal_html, corr_html, sesion_html = build_mercado(hora, hora_str)

    stop_usd = round(CAPITAL * STOP_PCT_DIARIO, 2)
    meta_usd = round(CAPITAL * META_PCT_DIARIA, 2)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="600">
<title>GASPA v9 — {hora_str}</title>
<style>{get_css()}</style>
</head>
<body>

<!-- HEADER -->
<div class="top-header">
    <div class="header-left">
        <div class="sys-name">Sistema GASPA V9.0 · IC Markets · Cuenta Real</div>
        <div class="title">Dashboard <span style="color:var(--blue)">Mercado</span></div>
    </div>
    <div class="header-right">
        <div class="hora">⏰ {hora_str} UY</div>
        <div id="countdown"></div>
        {sesion_html}
    </div>
</div>

<!-- PARÁMETROS -->
<div class="params-bar">
    <div class="p-item">SL <span class="p-val">{SL_MULTIPLICADOR}×ATR</span></div>
    <div class="p-item">TP <span class="p-val">{TP_MULTIPLICADOR}×ATR</span></div>
    <div class="p-item">R/R <span class="p-val">1:{round(TP_MULTIPLICADOR/SL_MULTIPLICADOR,2)}</span></div>
    <div class="p-item">Capital <span class="p-val">${CAPITAL:,.2f}</span></div>
    <div class="p-item">Riesgo <span class="p-val">{RIESGO_PCT*100:.0f}% = ${round(CAPITAL*RIESGO_PCT,2)}/trade</span></div>
    <div class="p-item p-stop">&#x1F6D1; Stop −{STOP_PCT_DIARIO*100:.0f}% = −${stop_usd}/día</div>
    <div class="p-item p-go">&#x1F3AF; Meta +{META_PCT_DIARIA*100:.0f}% = +${meta_usd}/día</div>
    <div class="p-item">Lateral <span class="p-val">ADX &lt; {ADX_LATERAL} en todos</span></div>
</div>

<!-- PROTOCOLO PSICOLÓGICO -->
{psico_html}

<!-- CARDS ACTIVOS -->
<div class="cards-container">
{cards_html}
</div>

<!-- SECCIONES INFERIORES -->
<div class="bottom-sections">
    {dxy_html}
    {cal_html}
    {corr_html}
</div>

<div class="footer">
    GASPA v9.0 · {hora_str} UY · Enzo Gasperi · Mente fría, lote fijo, proceso consistente
</div>

<script>
(function(){{
    var secs = 600, el = document.getElementById('countdown');
    if(!el) return;
    var iv = setInterval(function(){{
        secs--;
        if(secs <= 0){{ clearInterval(iv); el.textContent = 'actualizando...'; return; }}
        var m = Math.floor(secs/60), s = secs % 60;
        el.textContent = 'próx. actualización: ' + m + ':' + (s<10?'0':'') + s;
    }}, 1000);
}})();
</script>
</body>
</html>"""

    with open(ARCHIVO_SALIDA, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\n✅ HTML generado: {ARCHIVO_SALIDA}')

    if os.path.exists(DRIVE_FOLDER):
        ruta = os.path.join(DRIVE_FOLDER, ARCHIVO_SALIDA)
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'✅ Sincronizado: {ruta}')
    else:
        print(f'⚠️  Carpeta no encontrada: {DRIVE_FOLDER}')

if __name__ == '__main__':
    generar()
