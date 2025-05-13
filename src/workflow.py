import numpy as np
import pandas as pd
import os
from config import data_path, result_path, dates
import pickle
import warnings
from multiprocessing import Pool, cpu_count
from svi_calibrator_slsqp import SVICalibrator as SVI1
from svi_calibrator_de import SVICalibrator as SVI2
from svi_calibrator_qe import SVICalibrator as SVI3
from plot_curves import plot_multi_fit, plot_svi_surface, plot_all_residuals
from evaluation_metrics import SVICalibrationEvaluator
from stability_analysis import SVIStabilityAnalyzer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
warnings.filterwarnings("ignore")
# Configure multiprocessing to avoid numerical library thread explosion
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


with open(f"{data_path}/data_packs.pkl", "rb") as f:
    data_packs = pickle.load(f)


def multi_fit_analysis(date: str, method: str = 'slsqp', epsilon: float = 1e-3, verbose: bool = False, show: bool = False):
    """
    Perform multi-maturity SVI calibration and evaluation.
    Parameters
    ---------- 
    date : str
        The date for which to perform the analysis.
    method : str
        The calibration method to use. Options are 'slsqp' or 'de' or 'qe'.
    epsilon : float
        The epsilon value for the SVI calibration.
    verbose : bool
        If True, print detailed output during calibration.
    show : bool
        If True, show the plots. Default is False.
    """
    # ---------------------------------------------
    # Load the data
    # ---------------------------------------------
    dict_date = data_packs[date]
    k_list = [dict_expiry['k'] for dict_expiry in dict_date.values()]
    market_variance_list = [dict_expiry['w'].values for dict_expiry in dict_date.values()]
    maturities = [dict_expiry['T'] for dict_expiry in dict_date.values()]
    k_min = np.min([np.min(k) for k in k_list])
    k_max = np.max([np.max(k) for k in k_list])

    # Create the directory for the results
    svi_path = os.path.join(result_path, f'svi_{method}', date)
    if not os.path.exists(svi_path):
        os.makedirs(svi_path)

    # --------------------------------------------
    # Calibrate the SVI parameters
    # --------------------------------------------
    
    # Initialize the calibrator
    if method == 'slsqp':
        calibrator = SVI1(epsilon=epsilon, verbose=verbose)
    elif method == 'de':
        calibrator = SVI2(epsilon=epsilon, verbose=verbose)
    elif method == 'qe':
        calibrator = SVI3(epsilon=epsilon, verbose=verbose)
    else:
        raise ValueError("Invalid method. Choose 'slsqp' or 'de' or 'qe'.")
    # Multiple maturities calibration
    print("Calibrating multiply maturities")
    params_list = calibrator.robust_calibrate_multi(k_list, market_variance_list)

    # Plot the fitted curves
    plot_multi_fit(
        svi_path,
        date,
        calibrator, 
        k_list, 
        market_variance_list,
        params_list=params_list,
        maturities=maturities,
        show=show
    )

    # Create volatility surface plot
    plot_svi_surface(
        svi_path,
        date,
        calibrator,
        params_list,
        maturities=maturities,
        k_range=(k_min-0.005, k_max+0.005),
        show=show
    )

    # Print calibrated parameters
    params_df = calibrator.get_parameters_dataframe(maturities)
    params_df.to_csv(f"{svi_path}/svi_parameters_{date}.csv")
    print("\nCalibrated SVI Parameters:")
    print(params_df)

    # ----------------------------------------------------------
    # Evaluation and stability analysis
    # ----------------------------------------------------------

    # Initialize evaluator and stability analyzer
    evaluator = SVICalibrationEvaluator(calibrator)
    stability_analyzer = SVIStabilityAnalyzer(calibrator)

    # 1. Evaluate calibration quality
    print("\n===== CALIBRATION QUALITY EVALUATION =====")
    quality_metrics = evaluator.evaluate_multi_maturity_quality(
        k_list, market_variance_list, params_list, maturities
    )

    # Display summary metrics
    print("\nCalibration Quality Metrics:")
    print(quality_metrics)
    # Save quality metrics to CSV
    quality_metrics.to_csv(f"{svi_path}/quality_metrics_{date}.csv")

    # 2. Plot residuals for all maturities
    print("\n===== RESIDUAL ANALYSIS (ALL MATURITIES) =====")
    all_residual_stats = plot_all_residuals(
        svi_path,
        date,
        evaluator,
        k_list,
        market_variance_list, 
        params_list, 
        maturities,
        subplots_per_row=4
    )

    # Display summary of residuals for all maturities
    residual_summary = pd.DataFrame.from_dict(all_residual_stats, orient='index')
    print("\nResidual Statistics Summary:")
    print(residual_summary)
    # Save residual statistics to CSV
    residual_summary.to_csv(f"{svi_path}/residual_statistics_{date}.csv")


    # 3. Parameter sensitivity analysis
    print("\n===== PARAMETER SENSITIVITY ANALYSIS =====")
    param_variability, param_cv, _ = stability_analyzer.parameter_sensitivity(
        k_list[0], market_variance_list[0], params_list[0], 
        perturbation=0.01, n_trials=20
    )

    print("\nParameter sensitivity to 1% data perturbation:")
    print(param_variability)
    print(f"\nParameter Coefficient of Variation (lower is better): {param_cv}")
    # Save parameter sensitivity to CSV
    param_variability.to_csv(f"{svi_path}/parameter_sensitivity_{date}.csv")

    # 4. Maturity stability analysis
    print("\n===== MATURITY STABILITY ANALYSIS =====")
    maturity_stability = stability_analyzer.maturity_stability(
        k_list, market_variance_list, params_list, maturities
    )

    print("\nParameter changes between consecutive maturities:")
    print(maturity_stability[['maturity', 'prev_maturity', 'a_rel_change', 'b_rel_change', 'rho_rel_change', 'm_rel_change', 'sigma_rel_change', 'param_stability']])
    # Save maturity stability to CSV
    maturity_stability.to_csv(f"{svi_path}/maturity_stability_{date}.csv")

    # 5. Overall stability score
    print("\n===== OVERALL STABILITY SCORE =====")
    stability_score, _, _ = stability_analyzer.compute_stability_score(
        k_list, market_variance_list, params_list, maturities
    )

    print("Stability scores (lower is better):")
    # for metric, value in stability_score.items():
    #     print(f"{metric}: {value:.6f}")
    # Convert stability_score dictionary to a DataFrame and print
    stability_df = pd.DataFrame([stability_score])
    print(stability_df.T.rename(columns={0: 'Stability Score'}))
    
    # Save stability scores to CSV
    stability_df.to_csv(f"{svi_path}/stability_scores_{date}.csv", index=False)

    # 6. General Summary
    print("\n===== General Summary =====")

    # Overall calibration quality
    avg_rmse = quality_metrics['rmse'].mean()
    avg_r2 = quality_metrics['r_squared'].mean()

    # Check for arbitrage
    has_butterfly_arbitrage = (quality_metrics['arbitrage_risk'] > 0).any()
    has_calendar_arbitrage = 'has_calendar_arbitrage' in quality_metrics and quality_metrics['has_calendar_arbitrage'].any()

    # Parameter stability
    overall_stability = stability_score['overall_stability']

    print(f"Average RMSE across all maturities: {avg_rmse:.6f}")
    print(f"Average R² across all maturities: {avg_r2:.6f}")
    print(f"Butterfly arbitrage present: {'Yes' if has_butterfly_arbitrage else 'No'}")
    print(f"Calendar arbitrage present: {'Yes' if has_calendar_arbitrage else 'No'}")
    print(f"Overall parameter stability score: {overall_stability:.6f} (lower is better)")

def process_task(date_method, epsilon=1e-3, verbose=False, show=False):
    """Simple wrapper function for multiprocessing"""
    date, method = date_method
    print(f"Processing: {date}, Method: {method}")
    multi_fit_analysis(date, method=method, epsilon=epsilon, verbose=verbose, show=show)
    print(f"Finished processing: {date}, Method: {method}")


if __name__ == "__main__":
    method_list = ['slsqp', 'de', 'qe']
    tasks = [(date, method) for method in method_list for date in dates]
    
    # Set up multiprocessing pool
    num_processes = min(cpu_count(), len(tasks))
    print(f"Starting multiprocessing with {num_processes} processes")
    
    # Run tasks in parallel
    with Pool(processes=num_processes) as pool:
        pool.map(process_task, tasks)
