# -*- coding: utf-8 -*-
"""
WaterSip-DF-TTMC: dynamic RH thresholds / BLH factors, with TTMC correction
"""
from Step_1_Dynamic_RH_thresholds_calculation import dynamic_rh_thresholds_calculation
from Step_2_Dynamic_BLH_factors_calculation import dynamic_blh_factors_calculation
from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_4_P_simulation import p_simulation
from Step_5_Calculate_moisture_contribution import calculate_moisture_contribution
from functions import get_algorithm_params
import time


def main_watersip_df_ttmc():
    """Run WaterSip-DF-TTMC."""
    print("Starting WaterSip-DF-TTMC...")

    method = 'WaterSip-DF-TTMC'
    params = get_algorithm_params(method)

    update_DF = params['update_df']
    update_P_simulation = params['update_p_simulation']
    update_temporary = params['update_temporary']

#---------------------------------------------------------------------------

    if update_DF:
        # Step 1: Dynamic RH thresholds calculation
        dynamic_rh_thresholds_calculation(method)
        # Step 2: Dynamic BLH factors calculation
        dynamic_blh_factors_calculation(method)

    # Step 4 (optional) is global and independent of the target region
    if update_P_simulation:
        # Step 4: P simulation
        p_simulation(method, 'on')

    if update_temporary:
        # Step 3: Filtering and tracking
        start_time = time.time()
        filtering_and_tracking(method, 'on')
        print(f'execution time: {time.time()-start_time:.4f} s')

    # Step 5: Calculate moisture contribution (with TTMC correction)
    calculate_moisture_contribution('on', 'on')

    print("WaterSip-DF-TTMC completed!")


if __name__ == "__main__":
    main_watersip_df_ttmc()
