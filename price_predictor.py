import numpy as np
import pandas as pd
import logging
from typing import Dict, List

# Machine Learning Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Time Series Analysis
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

from data_fetcher import CryptoDataFetcher

class CryptoPricePredictor:
    """
    Advanced cryptocurrency price prediction module
    """
    def __init__(self):
        """
        Initialize price prediction components
        """
        self.logger = logging.getLogger(__name__)
        self.data_fetcher = CryptoDataFetcher()
        
        # Seed for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

    def _prepare_data(
        self, 
        token: str, 
        lookback: int = 60, 
        forecast_horizon: int = 30
    ) -> Dict:
        """
        Prepare time series data for machine learning models
        
        Args:
            token: Cryptocurrency token
            lookback: Number of previous days to use for prediction
            forecast_horizon: Number of days to predict
        
        Returns:
            Dictionary with prepared data
        """
        # Fetch historical price data
        historical_data = self.data_fetcher.get_token_historical_data(token)
        
        # Normalize data
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(historical_data[['price']])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_prices) - lookback - forecast_horizon):
            X.append(scaled_prices[i:i+lookback])
            y.append(scaled_prices[i+lookback:i+lookback+forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler
        }

    def lstm_prediction(
        self, 
        token: str, 
        lookback: int = 60, 
        forecast_horizon: int = 30
    ) -> Dict:
        """
        LSTM-based price prediction
        
        Args:
            token: Cryptocurrency token
            lookback: Number of previous days to use for prediction
            forecast_horizon: Number of days to predict
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Prepare data
            data = self._prepare_data(token, lookback, forecast_horizon)
            
            # Build LSTM Model
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(forecast_horizon)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error'
            )
            
            # Train model
            history = model.fit(
                data['X_train'], 
                data['y_train'], 
                epochs=50, 
                batch_size=32, 
                validation_split=0.2, 
                verbose=0
            )
            
            # Predict
            predictions = model.predict(data['X_test'])
            
            # Inverse transform predictions
            predictions = data['scaler'].inverse_transform(predictions.reshape(-1, 1))
            actual = data['scaler'].inverse_transform(data['y_test'].reshape(-1, 1))
            
            # Compute metrics
            mse = mean_squared_error(actual, predictions)
            mae = mean_absolute_error(actual, predictions)
            r2 = r2_score(actual, predictions)
            
            return {
                'predictions': predictions,
                'actual': actual,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
            }
        
        except Exception as e:
            self.logger.error(f"LSTM Prediction Error for {token}: {e}")
            return {}

    def arima_prediction(
        self, 
        token: str, 
        forecast_horizon: int = 30
    ) -> Dict:
        """
        ARIMA-based time series prediction
        
        Args:
            token: Cryptocurrency token
            forecast_horizon: Number of days to predict
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Fetch historical data
            historical_data = self.data_fetcher.get_token_historical_data(token)
            
            # Prepare data for ARIMA
            model = ARIMA(historical_data['price'], order=(5,1,0))
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(steps=forecast_horizon)
            forecast_index = pd.date_range(
                start=historical_data.index[-1], 
                periods=forecast_horizon+1, 
                closed='right'
            )
            
            forecast_df = pd.DataFrame({
                'forecast': forecast,
                'lower_ci': model_fit.conf_int()[:, 0],
                'upper_ci': model_fit.conf_int()[:, 1]
            }, index=forecast_index)
            
            return {
                'forecast': forecast_df,
                'summary': model_fit.summary()
            }
        
        except Exception as e:
            self.logger.error(f"ARIMA Prediction Error for {token}: {e}")
            return {}

    def prophet_prediction(
        self, 
        token: str, 
        forecast_horizon: int = 30
    ) -> Dict:
        """
        Facebook Prophet-based time series prediction
        
        Args:
            token: Cryptocurrency token
            forecast_horizon: Number of days to predict
        
        Returns:
            Dictionary with prediction results
        """
        try:
            # Fetch historical data
            historical_data = self.data_fetcher.get_token_historical_data(token)
            
            # Prepare data for Prophet
            prophet_df = historical_data.reset_index().rename(
                columns={'timestamp': 'ds', 'price': 'y'}
            )
            
            # Create and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True
            )
            model.fit(prophet_df)
            
            # Generate future dates
            future = model.make_future_dataframe(periods=forecast_horizon)
            forecast = model.predict(future)
            
            return {
                'forecast': forecast,
                'model': model
            }
        
        except Exception as e:
            self.logger.error(f"Prophet Prediction Error for {token}: {e}")
            return {}

    def ensemble_prediction(
        self, 
        token: str, 
        forecast_horizon: int = 30
    ) -> Dict:
        """
        Ensemble prediction combining multiple models
        
        Args:
            token: Cryptocurrency token
            forecast_horizon: Number of days to predict
        
        Returns:
            Dictionary with ensemble prediction results
        """
        # Run individual predictions
        predictions = {
            'lstm': self.lstm_prediction(token, forecast_horizon=forecast_horizon),
            'arima': self.arima_prediction(token, forecast_horizon),
            'prophet': self.prophet_prediction(token, forecast_horizon)
        }
        
        # Ensemble logic (simple average)
        ensemble_forecast = sum([
            pred['forecast'] for pred in predictions.values() 
            if 'forecast' in pred
        ]) / len(predictions)
        
        return {
            'ensemble_forecast': ensemble_forecast,
        }

    def predict_price(
        self, 
        token: str, 
        method: str = 'ensemble', 
        horizon: int = 30
    ) -> Dict:
        """
        Universal price prediction method
        
        Args:
            token: Cryptocurrency token
            method: Prediction method (ensemble, lstm, arima, prophet)
            horizon: Number of days to predict
        
        Returns:
            Dictionary with prediction results
        """
        # Mapping of prediction methods
        prediction_methods = {
            'ensemble': self.ensemble_prediction,
            'lstm': self.lstm_prediction,
            'arima': self.arima_prediction,
            'prophet': self.prophet_prediction
        }
        
        # Validate method
        if method not in prediction_methods:
            raise ValueError(f"Unsupported prediction method: {method}")
        
        # Execute prediction
        try:
            prediction = prediction_methods[method](token, horizon)
            return prediction
        except Exception as e:
            self.logger.error(f"Prediction error for {token} using {method}: {e}")
            return {}

    def risk_assessment(
        self, 
        token: str, 
        prediction_method: str = 'ensemble'
    ) -> Dict:
        """
        Comprehensive risk assessment for cryptocurrency
        
        Args:
            token: Cryptocurrency token
            prediction_method: Prediction method to use
        
        Returns:
            Dictionary with risk metrics
        """
        # Fetch historical data
        historical_data = self.data_fetcher.get_token_historical_data(token)
        
        # Calculate volatility metrics
        returns = historical_data['price'].pct_change()
        
        # Risk metrics
        risk_metrics = {
            'historical_volatility': returns.std(),
            'max_drawdown': self._calculate_max_drawdown(historical_data['price']),
            'value_at_risk_95': np.percentile(returns, 5),
            'value_at_risk_99': np.percentile(returns, 1)
        }
        
        # Get predictions for additional context
        try:
            predictions = self.predict_price(token, method=prediction_method)
            
            # Prediction uncertainty
            if 'ensemble_forecast' in predictions:
                forecast = predictions['ensemble_forecast']
                risk_metrics['forecast_uncertainty'] = (
                    forecast['upper_ci'] - forecast['lower_ci']
                ).mean()
        except Exception as e:
            self.logger.warning(f"Prediction for risk assessment failed: {e}")
        
        return risk_metrics

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            prices: Price series
        
        Returns:
            Maximum drawdown percentage
        """
        cumulative_max = prices.cummax()
        drawdown = (prices - cumulative_max) / cumulative_max
        return drawdown.min()

def main():
    """
    Test the CryptoPricePredictor
    """
    logging.basicConfig(level=logging.INFO)
    
    predictor = CryptoPricePredictor()
    
    # Test prediction methods
    tokens_to_test = ['bitcoin', 'ethereum', 'cardano']
    prediction_methods = ['ensemble', 'lstm', 'arima', 'prophet']
    
    for token in tokens_to_test:
        print(f"\n--- Predictions for {token.capitalize()} ---")
        
        for method in prediction_methods:
            print(f"\nMethod: {method}")
            try:
                prediction = predictor.predict_price(token, method=method)
                
                # Basic output (customize based on prediction method)
                if method == 'ensemble':
                    print("Ensemble Forecast Overview:")
                    print(prediction.get('ensemble_forecast', "No forecast available").head())
                elif method == 'lstm':
                    metrics = prediction.get('metrics', {})
                    print("LSTM Metrics:")
                    print(f"MSE: {metrics.get('mse', 'N/A')}")
                    print(f"MAE: {metrics.get('mae', 'N/A')}")
                    print(f"R2 Score: {metrics.get('r2', 'N/A')}")
                else:
                    print("Forecast generated successfully")
            
            except Exception as e:
                print(f"Prediction failed: {e}")
        
        # Risk Assessment
        print(f"\nRisk Assessment for {token.capitalize()}:")
        risk_metrics = predictor.risk_assessment(token)
        for metric, value in risk_metrics.items():
            print(f"{metric}: {value}")

if __name__ == "__main__":
    main()