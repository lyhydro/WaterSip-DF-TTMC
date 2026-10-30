# -*- coding: utf-8 -*-
"""
WaterSip: fixed RH threshold / BLH factor, no TTMC correction
"""
from Step_3_Filtering_and_tracking import filtering_and_tracking
from Step_5_Calculate_moisture_contribution import calculate_moisture_contribution
from functions import get_algorithm_params


def main_watersip():
    """Run WaterSip."""
    print("Starting WaterSip...")

    method = 'WaterSip'
    params = get_algorithm_params(method)

    update_temporary = params['update_temporary']

#---------------------------------------------------------------------------

    if update_temporary:
        # Step 3: Filtering and tracking
        filtering_and_tracking(method, 'off')

    # Step 5: Calculate moisture contribution
    calculate_moisture_contribution('off', 'off')

    print("WaterSip completed!")


if __name__ == "__main__":
    main_watersip()
