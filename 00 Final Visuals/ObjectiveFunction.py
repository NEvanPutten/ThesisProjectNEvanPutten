import numpy as np
from math import *

from Ac_constants import AcConstants

from AVL_dir.AVL_sim import AVL

AC = AcConstants()

def geometric_calc(design_vector):

    ###################### Initialize Variables from design vector ######################
    # Design vector variables
    wing_span = design_vector[0]
    chord_root = design_vector[1]
    chord_kink = design_vector[2]
    chord_tip = design_vector[3]
    sweep_le = design_vector[4]
    yoffset_kink = design_vector[5]
    zoffset_tip = design_vector[6]
    twist = design_vector[7]
    thickness_chord = design_vector[8]

    # Geometric transformations
    fuselage_width_half = AC.fuselage_width / 2  # Distance from centerline to root chord,  Total width 1.24
    sweep_le_rads = np.radians(sweep_le) #Convert to radians
    yoffset_tip = (wing_span / 2) - yoffset_kink #Distance from kink to tip
    # semi_span = (wing_span / 2) - fuselage_width_half #length of single exposed wing
    inboard_span = yoffset_kink - fuselage_width_half #length of inboard section wing, excluding fuselage
    chord_fuselage = chord_root + ((chord_root - chord_kink) / inboard_span) * (fuselage_width_half)


    ###################### Surface Area Panels ######################
    area_covered = ((chord_fuselage + chord_root) / 2) * (AC.fuselage_width) # entire fuselage
    area_inboard_covered = ((chord_fuselage + chord_kink) / 2) * (yoffset_kink * 2) #both wings
    area_outboard = ((chord_kink + chord_tip) / 2) * (yoffset_tip * 2) #both wings
    area_total = (area_inboard_covered + area_outboard)   # both wings


    ###################### Determine Equivalent Wing ######################
    #Equivalent wing params # Appendix D gudmundsson page 11
    #AR stays consistent

    ar_wing = (wing_span ** 2) / area_total

    area_weighted = (wing_span/area_total) * ((chord_fuselage* area_inboard_covered/2 + chord_kink * area_outboard/2)  + (chord_kink * area_inboard_covered/2 + chord_tip * area_outboard/2) ) #TODO area inboard on second parT?
    chord_fuselage_eq = (2/area_weighted) * (chord_fuselage* area_inboard_covered/2 + chord_kink * area_outboard/2)
    chord_tip_eq = (2/area_weighted) * (chord_kink * area_inboard_covered/2 + chord_tip * area_outboard/2)
    taper_ratio_eq = chord_tip_eq / chord_fuselage_eq
    sweep_le_eq = (2/area_total) * (sweep_le_rads * area_inboard_covered/2 + sweep_le_rads * area_outboard/2)
    sweep_quart_eq = np.arctan(tan(sweep_le_eq) + chord_fuselage_eq/(2*wing_span)*(taper_ratio_eq-1)) # quarter chord sweep of the equivalent wing in rads


    ###################### Determine wing weight ######################
    # Convert To Imperial
    dyn_pressure= 0.5 * AC.rho * (AC.velocity**2)
    dyn_pressure_cruise_lbf =  0.020885434233297 * dyn_pressure # lbf/ft2 #make sure its at cruise if flight conditions are altered
    total_surface_feet = area_total * (3.2808**2)
    gross_mass_lbs = AC.mass_total_ref * 2.2046#lbs
    weight_fuel_lbf = AC.mass_fuel * 2.2046# lbs

    # Calculations
    wing_mass_pounds = AC.correction_factor_raymer * 0.036 * (total_surface_feet**0.758) * (weight_fuel_lbf ** 0.0035)* ((ar_wing / (np.cos(sweep_quart_eq)**2)) ** 0.6) \
                          * (dyn_pressure_cruise_lbf**0.006) * (taper_ratio_eq** 0.04) * (((100*thickness_chord) / (np.cos(sweep_quart_eq))) ** -0.3) \
                          * ((AC.load_factor*gross_mass_lbs)**0.49)# [lbf]  #Snorri page 142, Raymer

    #Updating mass components
    wing_mass = wing_mass_pounds * 0.453592 # [lbf] to []kg]#should be about 20% of empty weight and 13% MTOW
    print('wing mass:', wing_mass)
    total_mass = AC.mass_body + wing_mass
    weight_force = total_mass * 9.81 # total weight of aircraft
    mass_empty = (AC.mass_empty_ref - AC.mass_wing_ref) + wing_mass
    wing_mass_percentage = (wing_mass/mass_empty)*100
    print('wing mass percentage',wing_mass_percentage)

    ###################### Determine Fuel Volume ######################
    # Volume only for outboard sections(no fuselage covered wing)
    # determine area of each chord
    area_root = np.pi * 0.3 * chord_root * 0.5 * ( thickness_chord * chord_root /2) + (
                (np.pi * 0.7 * chord_root * 0.5 * ( thickness_chord * chord_root /2)) + (
                    thickness_chord * chord_root * 0.7 * chord_root / 2)) / 2
    area_kink = np.pi * 0.3 * chord_kink * 0.5 * ( thickness_chord * chord_kink /2) + (
                (np.pi * 0.7 * chord_kink * 0.5 * ( thickness_chord * chord_kink /2)) + (
                    thickness_chord * chord_kink * 0.7 * chord_kink / 2)) / 2
    area_tip = np.pi * 0.3 * chord_tip * 0.5 * ( thickness_chord * chord_tip /2) + (
                (np.pi * 0.7 * chord_tip * 0.5 * ( thickness_chord * chord_tip /2)) + (
                    thickness_chord * chord_tip * 0.7 * chord_tip / 2)) / 2

    # Determining wing volume of inboard and outboard sections(both wings).
    wing_volume_inboard = 2 * (inboard_span) / 3 * (area_root + area_kink + np.sqrt(area_root * area_kink))
    wing_volume_outboard = 2 * (yoffset_tip) / 3 * (area_tip + area_kink + np.sqrt(area_tip * area_kink))
    wing_volume = wing_volume_outboard + wing_volume_inboard


    ###################### Determine Wing Loading ######################
    # Wing Loading only for outboard sections(no fuselage covered wing)
    area_exposed = area_total - area_covered
    wing_loading = total_mass / area_exposed

    return  area_total, weight_force, wing_mass_percentage, wing_volume, chord_fuselage, wing_loading,  area_inboard_covered, area_outboard

def empirical_profile_drag(design_vector, chord_fuselage, area_inboard_covered, area_outboard ):
    wing_span = design_vector[0]
    chord_root = design_vector[1]
    chord_kink = design_vector[2]
    chord_tip = design_vector[3]
    sweep_le = design_vector[4]
    yoffset_kink = design_vector[5]
    zoffset_tip = design_vector[6]
    twist = design_vector[7]
    thickness_chord = design_vector[8]


    temp = AC.temp #K at 5334m
    rho = AC.rho  # Kg/m^3
    velocity = AC.velocity # m/s (cruise speed of reference aircraft)
    #Dimension transformations

    yoffset_tip = (wing_span / 2) - yoffset_kink #Distance from kink to tip
    wetted_area_boost = 1.1
    sweep_le_rads = np.radians(sweep_le) #Convert to radians
    area_total = (area_inboard_covered + area_outboard)   # both wings
    area_wetted_inboard = wetted_area_boost * area_inboard_covered * 2  # includes fuselage page 683
    area_wetted_outboard = wetted_area_boost * area_outboard * 2
    # area_wetted_total = area_wetted_outboard + area_wetted_inboard

    # Flight conditions calculations
    nu = 1.458 * (temp ** 1.5) * (1 / (temp + 110.4)) * 10 ** (-6)  # N*s/m^2
    v_sound = (1.4 * 286.9 * temp) ** 0.5  # speed of sound at altitude

    Mach = velocity / v_sound  # Mach number of reference aircraft
    Re_fuselage = rho * velocity * chord_fuselage / nu  # at the fuselage
    Re_kink = rho * velocity * chord_kink / nu
    Re_tip = rho * velocity * chord_tip / nu
    dyn_pressure = 0.5 * rho * (velocity ** 2)  # kg/(m*s^2)

    ###################### Wing sweep inboard / outboard panels ######################
    x_te_inboard= -(chord_fuselage - yoffset_kink * np.tan(sweep_le_rads) - chord_kink) #X position of TE kink in rad
    sweep_te_inboard = np.arctan(x_te_inboard/yoffset_kink)
    sweep_inboard = (sweep_le_rads - sweep_te_inboard) * 0.75 + sweep_te_inboard # Quarter Chord
    x_te_outboard = -(chord_kink - yoffset_tip * np.tan(sweep_le_rads) - chord_tip) # X position of TE tip
    sweep_te_outboard = np.arctan(x_te_outboard/yoffset_tip)
    sweep_outboard = (sweep_le_rads - sweep_te_outboard) * 0.75 + sweep_te_outboard #Quarter Chord in rad


    ###################### Friction coeffient airfoils ######################
    # Root is located at the fuselage symmerty line, profile drag includes fuselage covered section
    transition_upper = 0.3
    transition_lower = 0.6

    XO_Croot_upper = 36.9 * (transition_upper ** (0.625)) * (1 / Re_fuselage) ** (0.375)
    Cf_root_upper = (0.074 / ((Re_fuselage) ** (0.2))) * (1 - (transition_upper - XO_Croot_upper)) ** (0.8)
    XO_Croot_lower = 36.9 * (transition_lower ** (0.625)) * (1 / Re_fuselage) ** (0.375)
    Cf_root_lower = (0.074 / ((Re_fuselage) ** (0.2))) * (1 - (transition_lower - XO_Croot_lower)) ** (0.8)
    Cf_root = (Cf_root_upper + Cf_root_lower) / 2

    XO_Ckink_upper = 36.9 * (transition_upper ** (0.625)) * (1 / Re_kink) ** (0.375)
    Cf_kink_upper = (0.074 / ((Re_kink) ** (0.2))) * (1 - (transition_upper - XO_Ckink_upper)) ** (0.8)
    XO_Ckink_lower = 36.9 * (transition_lower ** (0.625)) * (1 / Re_kink) ** (0.375)
    Cf_kink_lower = (0.074 / ((Re_kink) ** (0.2))) * (1 - (transition_lower - XO_Ckink_lower)) ** (0.8)
    Cf_kink = (Cf_kink_upper + Cf_kink_lower) / 2

    if Re_tip == 0:
        XO_Ctip = 0
        Cf_tip = 0
    else:
        XO_Ctip_upper = 36.9 * (transition_upper ** (0.625)) * (1 / Re_tip) ** (0.375)
        Cf_tip_upper = (0.074 / ((Re_tip) ** (0.2))) * (1 - (0.5 - XO_Ctip_upper)) ** (0.8)
        XO_Ctip_lower = 36.9 * (transition_lower ** (0.625)) * (1 / Re_tip) ** (0.375)
        Cf_tip_lower = (0.074 / ((Re_tip) ** (0.2))) * (1 - (0.5 - XO_Ctip_lower)) ** (0.8)
        Cf_tip = (Cf_tip_upper + Cf_tip_lower) / 2

    # Friction coefficient of each panel
    Cf_inboard = (Cf_root + Cf_kink) / 2  # average skin friction coefficient
    Cf_outboard = (Cf_kink + Cf_tip) / 2  # average skin friction coefficient

    # Total skin friction drag cofficient
    CDf_inboard = Cf_inboard * area_wetted_inboard / area_total
    CDf_outboard = Cf_outboard * area_wetted_outboard / area_total

    if Mach >= 0.2:
        compr_inboard = 1.34 * (Mach ** 0.18) * (np.cos(sweep_inboard) ** 0.28)
        compr_outboard = 1.34 * (Mach ** 0.18) * (np.cos(sweep_outboard) ** 0.28)

    else:
        compr_inboard = 1
        compr_outboard = 1

    # Form factor representing the profile drag of the entire wing.
    FF_inboard = compr_inboard * (1 + (0.6 / 0.25) * thickness_chord + 100 * thickness_chord ** 4)
    FF_outboard = compr_inboard * (1 + (0.6 / 0.25) * thickness_chord + 100 * thickness_chord ** 4)

    # Total profile drag of the wing
    CDp_empirical = CDf_inboard * FF_inboard + CDf_outboard * FF_outboard  # raymer method

    return CDp_empirical, dyn_pressure, Mach




def objective(design_vector):

    #Initialize Variables
    wing_span = design_vector[0]
    chord_root = design_vector[1]
    chord_kink = design_vector[2]
    chord_tip = design_vector[3]
    sweep_le = design_vector[4]
    yoffset_kink = design_vector[5]
    zoffset_tip = design_vector[6]
    twist = design_vector[7]
    thickness_chord = design_vector[8]

    ###################### GEOMETRY Constraint CHECK ######################
    # Intitialize geometric bounds
    kink_ub = AC.kink_ub
    kink_lb = AC.kink_lb
    chord_kink_lb = 0.50 * chord_root

    # Setup the geometric constraints
    if yoffset_kink - (AC.fuselage_width/2) > kink_ub * 0.5 * (wing_span - AC.fuselage_width):
        return 1.001, 0, 0, 0, 0

    if yoffset_kink - (AC.fuselage_width/2) < kink_lb * 0.5 * (wing_span - AC.fuselage_width):
        return 1.002, 0, 0, 0 , 0

    if chord_root < chord_kink or chord_kink < chord_tip:
        return 1.003, 0, 0,0 , 0

    if chord_kink_lb > chord_kink:
        return 1.004 , 0 , 0, 0 , 0

    ###################### Calculations for wing geometry and weight ######################
    area_total, weight_force, wing_mass_percentage, wing_volume, chord_fuselage, wing_loading,  area_inboard_covered, area_outboard = geometric_calc(design_vector)

    # Wing mass and volume bounds
    wing_mass_lb = AC.wing_mass_lb
    wing_mass_ub = AC.wing_mass_ub
    wing_volume_percentage = (weight_force/9.81) / AC.mass_total_ref # Scaling factor to increase required volume based on total mass increase
    wing_volume_product = AC.wing_volume_ref * wing_volume_percentage
    wing_mass_constr = 0
    wing_volume_constr   = 0
    wing_volume_product_constr = 0
    wing_loading_constr = 0

    print('Weight constraint:', wing_mass_percentage, wing_mass_ub)
    print('Volume constraint:', wing_volume, wing_volume_product)
    print('Loading constraint:', wing_loading,AC.wing_loading_ref * 1.1 )

    if wing_mass_percentage > wing_mass_ub or wing_mass_percentage < wing_mass_lb:
        CL_CDreturn =  1.005
        wing_mass_constr = wing_mass_percentage

    if  wing_volume < wing_volume_product:
        CL_CDreturn = 1.006
        wing_volume_constr = wing_volume
        wing_volume_product_constr = wing_volume_product

    if wing_loading  >  AC.wing_loading_ref * 1.1:
        CL_CDreturn =  1.007
        wing_loading_constr = wing_loading

    if wing_mass_constr > 0 or wing_volume_constr > 0 or wing_loading_constr > 0:
        return CL_CDreturn, wing_mass_constr, wing_volume_constr, wing_volume_product_constr,  wing_loading_constr

    ###################### Empirical profile drag calculations ######################
    CDp_empirical, dyn_pressure, Mach = empirical_profile_drag(design_vector, chord_fuselage, area_inboard_covered, area_outboard)
    CL_cruise = round(weight_force / (dyn_pressure * area_total), 5)


    ###################### Empirical profile drag calculations ######################
    alpha_cruise, CL, CDi = AVL(design_vector, area_total, Mach, chord_fuselage, CL_cruise)

    if alpha_cruise == -2:
        return 1e-8, 0, 0, 0, 0  # AVL CLfailed
    if CL ==  -2:
        return 1e-8, 0, 0, 0, 0  # AVL CLfailed
    if CDi == 1:
        return 1e-8, 0, 0, 0, 0  # AVL CDi failed

    CLCD_return = CL_cruise / (CDi + CDp_empirical)

    return CLCD_return, 0, 0, 0, 0


