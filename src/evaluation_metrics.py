import numpy as np
import pandas as pd


class SVICalibrationEvaluator:
    """
    Class to evaluate the quality and stability of SVI model calibration
    """
    
    def __init__(self, calibrator):
        """Initialize with a calibrator instance"""
        self.calibrator = calibrator
    
    def evaluate_fit_quality(self, k, market_variance, params, atm_range=0.05, wing_threshold=0.2):
        """
        Evaluate the quality of SVI calibration fit using multiple metrics
        
        Parameters:
        -----------
        k : array-like
            Log-moneyness values
        market_variance : array-like
            Market implied total variance values
        params : tuple
            Calibrated SVI parameters (a, b, rho, m, sigma)
        atm_range : float
            Range around 0 to consider as at-the-money (ATM)
        wing_threshold : float
            Threshold to define wing regions (abs(k) > wing_threshold)
            
        Returns:
        --------
        dict
            Dictionary of quality metrics
        """
        # Get model values
        model_variance = self.calibrator.svi_raw(k, params)
        
        # Calculate residuals
        residuals = model_variance - market_variance
        
        # Create masks for different regions
        atm_mask = np.abs(k) <= atm_range
        left_wing_mask = k < -wing_threshold
        right_wing_mask = k > wing_threshold
        
        # Calculate basic error metrics
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        max_abs_error = np.max(np.abs(residuals))
        
        # Calculate R-squared
        ss_total = np.sum((market_variance - np.mean(market_variance))**2)
        ss_residual = np.sum(residuals**2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Calculate region-specific metrics
        if sum(atm_mask) > 0:
            atm_rmse = np.sqrt(np.mean(residuals[atm_mask]**2))
            atm_bias = np.mean(residuals[atm_mask])
        else:
            atm_rmse = np.nan
            atm_bias = np.nan
            
        if sum(left_wing_mask) > 0:
            left_wing_rmse = np.sqrt(np.mean(residuals[left_wing_mask]**2))
        else:
            left_wing_rmse = np.nan
            
        if sum(right_wing_mask) > 0:
            right_wing_rmse = np.sqrt(np.mean(residuals[right_wing_mask]**2))
        else:
            right_wing_rmse = np.nan
        
        # Calculate wing asymmetry measure
        if not np.isnan(left_wing_rmse) and not np.isnan(right_wing_rmse):
            wing_asymmetry = abs(left_wing_rmse - right_wing_rmse) / max(left_wing_rmse, right_wing_rmse)
        else:
            wing_asymmetry = np.nan
        
        # Calculate arbitrage-free quality measure
        # g_function should be positive everywhere for no butterfly arbitrage
        k_fine = np.linspace(min(k)-0.2, max(k)+0.2, 200)
        g_values = self.calibrator.g_function(k_fine, params)
        min_g = np.min(g_values)
        arbitrage_risk = max(0, -min_g)  # Positive if arbitrage exists
        
        # Create results dictionary
        results = {
            'rmse': rmse,
            'mae': mae,
            'max_abs_error': max_abs_error,
            'r_squared': r_squared,
            'atm_rmse': atm_rmse,
            'atm_bias': atm_bias,
            'left_wing_rmse': left_wing_rmse,
            'right_wing_rmse': right_wing_rmse,
            'wing_asymmetry': wing_asymmetry,
            'arbitrage_risk': arbitrage_risk
        }
        
        return results

    def _evaluate_calendar_spread_quality(self, k_list, params_list, maturities):
        """Evaluate calendar spread quality between adjacent maturities"""
        n_maturities = len(params_list)
        
        # Get common k range
        k_min = min(np.min(k) for k in k_list)
        k_max = max(np.max(k) for k in k_list)
        k_test = np.linspace(k_min, k_max, 200)
        
        # Initialize results
        min_diff = np.zeros(n_maturities)
        has_arbitrage = np.zeros(n_maturities, dtype=bool)
        arbitrage_severity = np.zeros(n_maturities)
        
        # First maturity has no previous maturity to compare with
        min_diff[0] = np.nan
        has_arbitrage[0] = False
        arbitrage_severity[0] = np.nan
        
        # Check each pair of adjacent maturities
        for i in range(1, n_maturities):
            current_var = self.calibrator.svi_raw(k_test, params_list[i])
            prev_var = self.calibrator.svi_raw(k_test, params_list[i-1])
            diff = current_var - prev_var
            min_diff[i] = np.min(diff)
            has_arbitrage[i] = min_diff[i] < 0
            arbitrage_severity[i] = abs(min(0, min_diff[i]))
        
        return {
            'min_calendar_diff': min_diff,
            'has_calendar_arbitrage': has_arbitrage,
            'calendar_arbitrage_severity': arbitrage_severity
        }
    
    def evaluate_multi_maturity_quality(self, k_list, market_variance_list, params_list, maturities=None, atm_range=0.05, wing_threshold=0.2):
        """
        Evaluate the quality of SVI calibration across multiple maturities
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing quality metrics for each maturity
        """
        if maturities is None:
            maturities = [f"T{i+1}" for i in range(len(k_list))]
        
        # Initialize results storage
        metrics_dict = {}
        
        # Evaluate each maturity
        for i, (k, market_var, params) in enumerate(zip(k_list, market_variance_list, params_list)):
            maturity_key = f"{maturities[i]:.8f}" if isinstance(maturities[i], float) else str(maturities[i])
            metrics = self.evaluate_fit_quality(k, market_var, params, atm_range, wing_threshold)
            metrics_dict[maturity_key] = metrics
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        metrics_df.index.name = 'Maturity'
        
        # Calculate calendar spread metrics
        if len(params_list) > 1:
            calendar_metrics = self._evaluate_calendar_spread_quality(k_list, params_list, maturities)
            # Add calendar metrics to the dataframe
            for col, values in calendar_metrics.items():
                metrics_df[col] = values
        
        return metrics_df
