# -*- coding: utf-8 -*-
"""
WaterSip-TTMC: fixed RH threshold / BLH factor, with TTMC correction
"""
from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_4_P_simulation import p_simulation
from Step_5_Calculate_moisture_contribution import calculate_moisture_contribution
from functions import get_algorithm_params


def main_watersip_ttmc():
    """Run WaterSip-TTMC."""
    print("Starting WaterSip-TTMC ...")

    method = 'WaterSip-TTMC'
    params = get_algorithm_params(method)

    update_P_simulation = params['update_p_simulation']
    update_temporary = params['update_temporary']

#---------------------------------------------------------------------------

    # Step 4 (optional) is global and independent of the target region
    if update_P_simulation:
        # Step 4: P simulation
        p_simulation(method, 'off')

    if update_temporary:
        # Step 3: Filtering and tracking
        filtering_and_tracking(method, 'off')

    # Step 5: Calculate moisture contribution (with TTMC correction)
    calculate_moisture_contribution('off', 'on')

    print("WaterSip-TTMC completed!")


if __name__ == "__main__":
    main_watersip_ttmc()
