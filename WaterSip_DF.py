# -*- coding: utf-8 -*-
"""
WaterSip-DF: dynamic RH thresholds / BLH factors, no TTMC correction
"""
from Step_1_Dynamic_RH_thresholds_calculation import dynamic_rh_thresholds_calculation
from Step_2_Dynamic_BLH_factors_calculation import dynamic_blh_factors_calculation
from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_5_Calculate_moisture_contribution import calculate_moisture_contribution
from functions import get_algorithm_params


def main_watersip_df():
    """Run WaterSip-DF."""
    print("Starting WaterSip-DF...")

    method = 'WaterSip-DF'
    params = get_algorithm_params(method)

    update_DF = params['update_df']
    update_temporary = params['update_temporary']

#---------------------------------------------------------------------------

    if update_DF:
        # Step 1: Dynamic RH thresholds calculation
        dynamic_rh_thresholds_calculation(method)
        # Step 2: Dynamic BLH factors calculation
        dynamic_blh_factors_calculation(method)

    if update_temporary:
        # Step 3: Filtering and tracking
        filtering_and_tracking(method, 'on')

    # Step 5: Calculate moisture contribution
    calculate_moisture_contribution('on', 'off')

    print("WaterSip-DF completed!")


if __name__ == "__main__":
    main_watersip_df()
