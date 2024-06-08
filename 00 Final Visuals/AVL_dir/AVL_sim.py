import os, sys
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Ac_constants import AcConstants

AC = AcConstants()

def AVL(design_vector, area_total, Mach, chord_fuselage, CL_cruise):
    # CL_cruise !!!

    # Generate geometry file for AVL
    geometry(design_vector, area_total, Mach, chord_fuselage)

    # Generate run file for AVL
    cases(Mach, CL_cruise)

    # Generate AVL Instructions
    # runs = AcConstants.runs
    instructions()

    # Running AVL simulations
    avl_output_file = r"AVL_dir\avl_output.txt"
    command = r"AVL_dir\avl.exe < AVL_dir/instructions.txt > {}".format(avl_output_file)
    os.system(command)

    # Extract results


    with open(r"AVL_dir/forces.txt", 'r') as fp:
        lines = len(fp.readlines())

        if lines > 0:
            alpha_cruise = np.loadtxt('AVL_dir/forces.txt', skiprows=17 , usecols=2, max_rows=1)
            CL = np.loadtxt('AVL_dir/forces.txt', skiprows=23 , usecols=2, max_rows=1)
            CDi = np.loadtxt('AVL_dir/forces.txt', skiprows=24 , usecols=2, max_rows=1)
        else:
            alpha_cruise = -2
            CL = -2
            CDi = 1

    return alpha_cruise, CL, CDi


def cases(Mach, CL_cruise):

    velocity = AC.velocity
    density = AC.rho
    Mach = Mach

    if os.path.isfile("AVL_dir/cases.run"):
        f = open("AVL_dir/cases.run", "w")
    else:
        f = open("AVL_dir/cases.run", "w")

    f.writelines([' ---------------------------------------------\n',
                 f' Run case  {1}:  cruise condition\n',
                 ' \n',
                 f' alpha        ->  CL          =   {CL_cruise}\n',
                 ' beta         ->  beta        =   0.00000\n',
                 ' pb/2V        ->  pb/2V       =   0.00000\n',
                 ' qc/2V        ->  qc/2V       =   0.00000\n',
                 ' rb/2V        ->  rb/2V       =   0.00000\n',
                 ' \n',
                 f' alpha     =   0.00000     deg\n',
                 ' beta      =   0.00000     deg\n',
                 ' pb/2V     =   0.00000\n',
                 ' qc/2V     =   0.00000\n',
                 ' rb/2V     =   0.00000\n',
                 f' CL        =   {CL_cruise}\n',
                 ' CDo       =   0.00000\n',
                 ' bank      =   0.00000 deg\n',
                 ' elevation =   0.00000 deg\n',
                 ' heading   =   0.00000 deg\n',
                 f' Mach      =   {Mach:.8f}\n',
                 f' velocity  =   {velocity:.8f} Lunit/Tunit\n',
                 f' density   =   {density:.8f} Munit/Lunit^3\n',
                 ' grav.acc. =   9.81000 Lunit/Tunit^2\n',
                 ' turn_rad. =   0.00000 Lunit\n',
                 ' load_fac. =   0.00000\n',
                 ' X_cg      =   0.00000 Lunit\n',
                 ' Y_cg      =   0.00000 Lunit\n',
                 ' Z_cg      =   0.00000 Lunit\n',
                 ' mass      =   1.00000 Munit\n',
                 ' Ixx       =   1.00000 Munit-Lunit^2\n',
                 ' Iyy       =   1.00000 Munit-Lunit^2\n',
                 ' Izz       =   1.00000 Munit-Lunit^2\n',
                 ' Ixy       =   0.00000 Munit-Lunit^2\n',
                 ' Iyz       =   0.00000 Munit-Lunit^2\n',
                 ' Izx       =   0.00000 Munit-Lunit^2\n',
                 ' visc CL_a =   0.00000\n',
                 ' visc CL_u =   0.00000\n',
                 ' visc CM_a =   0.00000\n',
                 ' visc CM_u =   0.00000\n'])
    f.close()

    return

def geometry(design_vector,area_total, Mach, chord_fuselage):
    name_run = 'TestRun'

    wing_span = design_vector[0]
    chord_root = design_vector[1]
    chord_kink = design_vector[2]
    chord_tip = design_vector[3]
    sweep_le = design_vector[4]
    yoffset_kink = design_vector[5]
    zoffset_tip = design_vector[6]
    twist = design_vector[7]
    thickness_chord = design_vector[8]


    #Calculations
    sweep = np.radians(sweep_le)
    area_total = area_total
    X_root = (AC.fuselage_width/2) * np.tan(sweep)
    Y_root = (AC.fuselage_width/2)
    Z_root = ((AC.fuselage_width/2) / (wing_span/2) ) * zoffset_tip
    X_kink = yoffset_kink * np.tan(sweep)
    Y_kink = yoffset_kink
    Z_kink = (yoffset_kink / (wing_span/2) ) * zoffset_tip
    X_tip = X_kink + ((wing_span / 2) - yoffset_kink) * np.tan(sweep)
    Y_tip = wing_span / 2
    Z_tip = zoffset_tip

    twist_kink = ((yoffset_kink - AC.fuselage_width / 2) / ((wing_span-AC.fuselage_width)/2) ) * twist


    if os.path.isfile("AVL_dir/geometry.avl"):
        f = open("AVL_dir/geometry.avl", "w")
    else:
        f = open("AVL_dir/geometry.avl", "w")


    f.writelines([f'{name_run}\n',
            '#Mach\n',
            f'{Mach: .8f} \n',
            '#IYsym   IZsym   Zsym\n',
            ' 0       0       0.0\n',
            '#Sref    Cref    Bref\n',
            f' {area_total:.8f}     {(area_total/wing_span):.8f}     {wing_span:.8f}\n',
            '#Xref    Yref    Zref\n',
            '0.00     0.0     0.0\n',
            '#\n',
            '#\n',
            '#====================================================================\n',
            'SURFACE \n',
            'Main Wing \n',
            '#Nchordwise  Cspace   Nspanwise   Sspace\n',
            '8            1.0       22         1.0\n',
            '#\n',
            'YDUPLICATE\n',
            '0.0\n',
            '#\n',
            'ANGLE\n',
            '0.0\n',
            '#-------------------------------------------------------------\n',
            'SECTION\n',
            '#Xle    Yle    Zle     Chord   Ainc  Nspanwise  Sspace\n',
            f'0.      0.     0.      {chord_fuselage :.8f}     0.0   0          0\n',
            '\n',
            'CLAF\n',
            f'{1.0 + 0.77 * thickness_chord}',
            '\n'
            '#-------------------------------------------------------------\n',
            'SECTION\n',
            '#Xle    Yle    Zle     Chord   Ainc  Nspanwise  Sspace\n',
            f'{X_root:.8f}      {Y_root:.8f}    {Z_root:.8f}      {chord_root :.8f}     0.0   0          0\n',
            '\n',
            'CLAF\n',
            f'{1.0 + 0.77*thickness_chord}',
            '\n'
            '#-------------------------------------------------------------\n',
            'SECTION\n',
            '#Xle    Yle    Zle     Chord   Ainc  Nspanwise  Sspace\n',
            f'{X_kink:.8f}     {Y_kink:.8f}    {Z_kink:.8f}        {chord_kink:.8f}    {twist_kink : .8f}   0          0\n',
            '\n',
            'CLAF\n',
            f'{1.0 + 0.77 * thickness_chord}',
            '\n'
            '#-------------------------------------------------------------\n',
            'SECTION\n',
            '#Xle    Yle    Zle     Chord   Ainc  Nspanwise  Sspace\n',
            f'{X_tip:.8f}     {Y_tip:.8f}    {Z_tip:.8f}        {chord_tip:.8f}    {twist : .8f}   0          0\n',
            '\n',
            'CLAF\n',
            f'{1.0 + 0.77 * thickness_chord}',
            '\n'
            '#====================================================================\n'])
    f.close()

    return

def instructions():

    if os.path.isfile(r"AVL_dir\instructions.txt"):
        f = open(r"AVL_dir\instructions.txt", "w")#Overwrite?
    else:
        f = open(r"AVL_dir\instructions.txt", "w")

    forces_file= r'AVL_dir\forces.txt'

    if os.path.isfile(r"AVL_dir\forces.txt"):
        f2 = open(r"AVL_dir\forces.txt", "w")#Overwrite?
        f2.close()


    f.writelines(['load\n',
                  'AVL_dir/geometry.avl\n'
                  'case\n',
                  'AVL_dir/cases.run\n',
                  'oper\n',
                  'x\n',
                  'FT \n',
                  f'{forces_file}\n',
                  'O\n',
                  '\n',
                  'q\n'])


    f.close()

    return
