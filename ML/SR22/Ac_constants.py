# Cirrus SR22
import numpy as np


class AcConstants:

    def __init__(self):
        aircraft_configs = ['SR22', 'DA50', 'ASK21','F50']
        self.aircraft_config = aircraft_configs[0] #SELECT BASE AIRCRAFT CONFIGURATION
        self.load_aircraft_data()

    def load_aircraft_data(self):
        #Store all starting stats of aircraft
        if self.aircraft_config ==  'SR22':
            # Wing starting dimensions
            self.fuselage_width = 1.24
            self.wing_span_start = 11.65
            self.chord_root_start = 1.54
            self.chord_kink_start = 1.145
            self.chord_tip_start = 0.75
            self.sweep_le_start = 2
            self.yoffset_kink_start = 3.225
            self.zoffset_tip_start = 0.4
            self.twist_start = 0
            self.thickness_chord_start = 0.196

            # Aircraft reference
            self.mass_total_ref = 1633  # 1633[kg] ref cr s22
            self.mass_empty_ref = 1067.8  # [kg] from reference
            self.mass_wing_ref = 156.4 #209 [kg] 20% from empty weight
            self.mass_body = self.mass_total_ref - self.mass_wing_ref
            self.mass_fuel = 220
            self.load_factor = 3.8 * 1.5
            self.wing_volume_ref = 1.90
            self.wing_loading_ref = 136
            self.correction_factor_raymer = 1#1.3358
            self.transfer_learning = 1


            #Get the flight conditions
            self.objective_start = 28.08
            self.temp = 253.6  # K at 5334m
            self.rho = 0.711  # Kg/m^3 density at cruise altitude
            self.velocity = 92.6  # [m/s] True air speed at cruise > 180 kts
            self.vel_max = 95.69 # > 186 kts

            # self.rho_ref = 0.711  # [kg/m^3] density at cruise altitude
            # self.EAS = self.vel_ref * np.sqrt(self.rho_ref / 1.225) #[-]
            # self.EAS_max = self.vel_max * np.sqrt(self.rho_ref / 1.225) #[-]
            # self.velocity = self.EAS / np.sqrt(self.rho / 1.225)  # m/s (cruise speed of reference aircraft)

            # Aircraft category : constraints and bounds
            self.upper_bound = [15, 2.5, 2, 1.5, 20, 5, 1, 5, 0.22]
            self.lower_bound = [8, 0.5, 0.5, 0.1, 0, 1.4, 0, -5, 0.11]
            self.kink_ub = 0.75
            self.kink_lb = 0.20
            self.wing_mass_lb = 13
            self.wing_mass_ub = 27

        elif self.aircraft_config == 'DA50':
            # Wing starting dimensions
            self.fuselage_width = 1.29
            self.wing_span_start = 13.41
            self.chord_root_start = 1.724
            self.chord_kink_start = 1.287
            self.chord_tip_start = 0.92
            self.sweep_le_start = 2
            self.yoffset_kink_start = 1.396 + self.fuselage_width / 2
            self.zoffset_tip_start = 0.564
            self.twist_start = 0
            self.thickness_chord_start = 0.137

            # Aircraft reference
            self.mass_total_ref = 1999
            self.mass_empty_ref = 1450  # [kg] from reference
            self.mass_wing_ref = 290  # [kg] 20% from empty weight
            self.mass_body = self.mass_total_ref - self.mass_wing_ref
            self.mass_fuel = 148
            self.load_factor = 3.8 * 1.5
            self.wing_volume_ref = 1.67
            self.wing_loading_ref = 137
            self.correction_factor_raymer =1.2364
            self.transfer_learning = 1

            #Get the flight conditions
            self.objective_start = 33.20
            self.temp = 260.41   # K at 5334m
            self.rho = 0.797 # Kg/m^3 density at cruise altitude
            self.velocity = 88  # [m/s] True air speed at cruise > 180 kts
            self.vel_max = 93 # > 186 kts

            # Aircraft category : constraints and bounds
            self.upper_bound = [15, 2.5, 2, 1.5, 20, 5, 1, 5, 0.22]
            self.lower_bound = [8, 0.5, 0.5, 0.1, 0, 1.4, 0, -5, 0.11]
            self.kink_ub = 0.75
            self.kink_lb = 0.20
            self.wing_mass_lb = 13
            self.wing_mass_ub = 28

        elif self.aircraft_config == 'ASK21':
            # Wing starting dimensions
            self.fuselage_width = 0.70
            self.wing_span_start = 17.0
            self.chord_root_start = 1.50
            self.chord_kink_start = 1.0
            self.chord_tip_start = 0.50
            self.sweep_le_start = 0
            self.yoffset_kink_start = 4.81 + self.fuselage_width/2
            self.zoffset_tip_start = 0.594  # 4deg dihidral
            self.twist_start = 0
            self.thickness_chord_start = (0.196 + 0.126) / 2

            # Aircraft reference
            self.mass_total_ref = 600
            self.mass_empty_ref = 401  # [kg] from reference
            self.mass_wing_ref = 198  # [kg] # per wing 50% empty weight
            self.mass_body = self.mass_total_ref - self.mass_wing_ref
            self.mass_fuel = 1
            self.load_factor = 6.5
            self.wing_volume_ref = 2.11
            self.wing_loading_ref = 35
            self.correction_factor_raymer =1.1450642
            self.transfer_learning = 1

            #Get the flight conditions
            # At cruise altitude of m (5000ft)
            self.objective_start = 26.50
            self.temp = 278.24   # K at 5334m
            self.rho = 1.056 # Kg/m^3 density at cruise altitude
            self.velocity = 49.90  # [m/s] True air speed at cruise
            self.vel_max =  77.68 #

            # Aircraft category : constraints and bounds
            self.upper_bound = [24, 2.5, 2, 1.5, 20, 10, 1, 5, 0.20]  # Changed max wing span and ykink
            self.lower_bound = [8, 0.5, 0.5, 0.1, 0, 1.4, 0, -5, 0.11]
            self.kink_ub = 0.75
            self.kink_lb = 0.20
            self.wing_mass_lb = 30  # changed
            self.wing_mass_ub = 55  # changed

        elif self.aircraft_config == 'F50':
            # Wing starting dimensions
            # http: // www.flugzeuginfo.net / acdata_php / acdata_fokker50_en.php
            self.fuselage_width = 2.7
            self.wing_span_start = 29
            self.chord_root_start = 3.3
            self.chord_kink_start = 2.36
            self.chord_tip_start = 1.43
            self.sweep_le_start = 3.75 #7.56
            self.yoffset_kink_start = 7.92
            self.zoffset_tip_start =  0.433 # 7deg
            self.twist_start = 0
            self.thickness_chord_start = 0.1795  #https://secure.simmarket.com/paob-fokker-50-fsx-(zn_4665).phtml

            # Aircraft reference
            self.mass_total_ref = 18990
            self.mass_empty_ref = 12570 #https://www.airliners.net/aircraft-data/fokker-50/218
            self.mass_wing_ref = 2514 #20% of empty weight
            self.mass_body = self.mass_total_ref - self.mass_wing_ref
            self.mass_fuel = 4080 #https://www.smartcockpit.com/docs/Fokker_50-Fuel.pdf
            self.load_factor = 2.5 * 1.5
            self.wing_volume_ref = 19.00
            self.wing_loading_ref = 305
            self.correction_factor_raymer = 1.472966
            self.transfer_learning = 1

            # Get the flight conditions
            # At cruise altitude of m (5000ft)
            self.objective_start = 34.91
            self.temp = 238.68  # K at 5334m
            self.rho = 0.55 # Kg/m^3 density at cruise altitude
            self.velocity = 147 # [m/s] True air speed at cruise
            self.vel_max = 156  #

            # Aircraft category : constraints and bounds
            self.upper_bound = [36, 6,6,3, 20, 12.5, 5, 5, 0.22]  # CHANGED to scale > class aircraft max span.
            self.lower_bound = [15, 1, 1, 0.5, 0, 1.35, 0, -5, 0.08]  # CHANGED to scale
            self.kink_ub= 0.75
            self.kink_lb = 0.20
            self.wing_mass_lb = 13
            self.wing_mass_ub = 27

        else:
            print('NO AIRCRAFT CONFIG OR FLIGHT CONDITION FOUND')

