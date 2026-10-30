# -*- coding: utf-8 -*-
"""WaterSip-DF-TTMC entry point."""
import sys

USAGE = 'USAGE: python main.py [watersip|watersip-df|watersip-ttmc|watersip-df-ttmc]'

if __name__ == "__main__":
    algorithm = sys.argv[1].lower() if len(sys.argv) > 1 else ''

    if algorithm == 'watersip':
        from WaterSip import main_watersip
        main_watersip()
    elif algorithm == 'watersip-df':
        from WaterSip_DF import main_watersip_df
        main_watersip_df()
    elif algorithm == 'watersip-ttmc':
        from WaterSip_TTMC import main_watersip_ttmc
        main_watersip_ttmc()
    elif algorithm == 'watersip-df-ttmc':
        from WaterSip_DF_TTMC import main_watersip_df_ttmc
        main_watersip_df_ttmc()
    else:
        print(USAGE)
