import numpy as np
import pandas as pd


class SVIStabilityAnalyzer:
    """
    Class to analyze the stability and robustness of SVI calibration
    """
    
    def __init__(self, calibrator):
        """Initialize with a calibrator instance"""
        self.calibrator = calibrator
    
    def parameter_sensitivity(self, k, market_variance, params, perturbation=0.01,  n_trials=10, random_seed=42):
        """
        Analyze parameter sensitivity by adding small perturbations to market data
        
        Parameters:
        -----------
        k : array-like
            Log-moneyness values
        market_variance : array-like
            Market implied total variance values
        params : tuple
            Original calibrated SVI parameters (a, b, rho, m, sigma)
        perturbation : float
            Standard deviation of perturbation relative to market variance
        n_trials : int
            Number of Monte Carlo trials
        random_seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        tuple
            (parameter_variability, param_cv, recalibration_results)
        """
        np.random.seed(random_seed)
        
        # Initialize storage for parameters
        param_names = ['a', 'b', 'rho', 'm', 'sigma']
        all_params = np.zeros((n_trials, len(params)))
        all_rmse = np.zeros(n_trials)
        
        # Run multiple trials with perturbed data
        for i in range(n_trials):
            # Create perturbed market data
            noise = np.random.normal(0, perturbation * np.mean(market_variance), len(market_variance))
            perturbed_variance = np.maximum(1e-6, market_variance + noise)  # Ensure positive
            
            # Recalibrate with perturbed data
            try:
                recal_params = self.calibrator.robust_calibration(k, perturbed_variance)
                all_params[i] = recal_params
                
                # Calculate RMSE of the recalibration on original data
                model_variance = self.calibrator.svi_raw(k, recal_params)
                residuals = model_variance - market_variance
                all_rmse[i] = np.sqrt(np.mean(residuals**2))
            except:
                # If calibration fails, use original parameters
                all_params[i] = params
                all_rmse[i] = np.nan
        
        # Calculate parameter statistics
        param_mean = np.mean(all_params, axis=0)
        param_std = np.std(all_params, axis=0)
        param_cv = param_std / np.abs(param_mean)  # Coefficient of variation
        
        # Calculate normalized parameter range (max-min)/original
        param_range = (np.max(all_params, axis=0) - np.min(all_params, axis=0)) / np.abs(params)
        
        # Aggregate results
        param_variability = pd.DataFrame({
            'original': params,
            'mean': param_mean,
            'std': param_std,
            'cv': param_cv,
            'norm_range': param_range
        }, index=param_names)
        
        # Store all recalibration results
        recalibration_results = {
            'parameters': all_params,
            'rmse': all_rmse
        }
        
        return param_variability, param_cv, recalibration_results
    
    def maturity_stability(self, k_list, market_variance_list, params_list, maturities=None):
        """
        Analyze stability of parameters across maturities
        
        Returns:
        --------
        pandas.DataFrame
            Parameter stability metrics across maturities
        """
        if maturities is None:
            maturities = np.arange(1, len(params_list) + 1)
        
        # Convert params_list to array
        params_array = np.array(params_list)
        
        # Calculate parameter changes between consecutive maturities
        param_changes = np.diff(params_array, axis=0)
        relative_changes = param_changes / np.abs(params_array[:-1])
        
        # Create parameter names
        param_names = ['a', 'b', 'rho', 'm', 'sigma']
        
        # Calculate stability metrics
        stability_metrics = {
            'maturity': maturities[1:],
            'prev_maturity': maturities[:-1]
        }
        
        for i, name in enumerate(param_names):
            stability_metrics[f'{name}_change'] = param_changes[:, i]
            stability_metrics[f'{name}_rel_change'] = relative_changes[:, i]
        
        # Add overall stability score
        stability_metrics['param_stability'] = np.sqrt(np.sum(relative_changes**2, axis=1))
        
        # Convert to DataFrame
        stability_df = pd.DataFrame(stability_metrics)
        
        return stability_df
    
    def compute_stability_score(self, k_list, market_variance_list, params_list, maturities=None):
        """
        Compute a comprehensive stability score for the calibration
        
        Returns:
        --------
        dict
            Stability scores and metrics
        """
        # Set default maturities if not provided
        if maturities is None:
            maturities = np.arange(1, len(params_list) + 1)
        
        # Calculate maturity stability
        maturity_stability = self.maturity_stability(k_list, market_variance_list, params_list, maturities)
        
        # Parameter sensitivity for first maturity
        param_variability, param_cv, _ = self.parameter_sensitivity(
            k_list[0], market_variance_list[0], params_list[0], perturbation=0.01, n_trials=20
        )
        
        # Calculate average parameter change between maturities
        avg_param_change = np.mean(maturity_stability['param_stability'])
        max_param_change = np.max(maturity_stability['param_stability'])
        
        # Parameter volatility across maturities
        param_vol = np.std(params_list, axis=0) / np.mean(np.abs(params_list), axis=0)
        
        # Combine metrics into stability score
        # Lower is better (more stable)
        stability_score = {
            'param_sensitivity': np.mean(param_cv),
            'avg_maturity_change': avg_param_change,
            'max_maturity_change': max_param_change,
            'param_volatility': np.mean(param_vol),
            'overall_stability': np.mean([np.mean(param_cv), avg_param_change, np.mean(param_vol)])
        }
        
        for i, name in enumerate(['a', 'b', 'rho', 'm', 'sigma']):
            stability_score[f'{name}_sensitivity'] = param_cv[i]
            stability_score[f'{name}_volatility'] = param_vol[i]
        
        return stability_score, maturity_stability, param_variability
