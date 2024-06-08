import os, sys

# from ObjectiveFunction import objective
import matplotlib.pyplot as plt
import numpy as np
from Ac_constants import AcConstants
import time
from Results import Results
import matplotlib.pyplot as plt
import scipy as sp
from ObjectiveFunction import objective
import glob

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# which analysis are needed to evaluate the wing?
# 1) Evaluation of wing performance:
# Standard CL_cruise/ CD calculation
# 2) Plotting of multiple planforms next to each other
# 3) XFOIL vs Empirical profile drag > at 3 deg
# 4)

AC = AcConstants()
res = Results()


def get_PSO():
    # get directory
    aircraft_config = 'SR22'
    directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
    results_PSO_list = os.listdir(directory_PSO)
    CL_CD_PSO = []
    design_vectors_PSO = []

    for f in range(len(results_PSO_list)):
        CL_CD = 0
        file_name = results_PSO_list[f]
        file_directory = directory_PSO + '/' + file_name

        with open(file_directory, 'r') as fp:
            lines = len(fp.readlines())
        if lines > 0:
            CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
            design_vector = []
            for dv in range(9):
                dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                design_vector.append(-dv_i)
        design_vectors_PSO.append(design_vector)
        CL_CD_PSO.append(CL_CD)

    sorted_CL_CD_PSO = sorted(CL_CD_PSO)
    best_CL_CD_PSO = sorted_CL_CD_PSO[-10:]
    best_CL_CD_PSO.reverse()
    best_idx_PSO = []
    best_design_PSO = []

    for idx in range(len(best_CL_CD_PSO)):
        best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

    for idx in range(len(best_CL_CD_PSO)):
        best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])
    return best_CL_CD_PSO, best_design_PSO


def points(design_vector, fuselage_width):
    wing_span = design_vector[0]
    chord_root = design_vector[1]
    chord_kink = design_vector[2]
    chord_tip = design_vector[3]
    sweep_le = design_vector[4]
    yoffset_kink = design_vector[5]
    zoffset_tip = design_vector[6]
    twist = design_vector[7]
    thickness_chord = design_vector[8]

    inboard_span = yoffset_kink - fuselage_width / 2  # length of inboard section wing, excluding fuselage
    chord_fuselage = chord_root + (chord_root - chord_kink) / inboard_span * (yoffset_kink - inboard_span)
    sweep = np.radians(design_vector[4])

    x_fuselageLE = 0
    y_fuselageLE = 0
    x_fuselageTE = 0
    y_fuselageTE = -chord_fuselage

    x_rootLE = (fuselage_width / 2)
    y_rootLE = -  (fuselage_width / 2) * np.tan(sweep)
    x_rootTE = (fuselage_width / 2)
    y_rootTE = -  (fuselage_width / 2) * np.tan(sweep) - chord_root

    x_kinkLE = yoffset_kink
    y_kinkLE = - yoffset_kink * np.tan(sweep)
    x_kinkTE = yoffset_kink
    y_kinkTE = - yoffset_kink * np.tan(sweep) - chord_kink

    x_tipLE = wing_span / 2
    y_tipLE = - wing_span / 2 * np.tan(sweep)
    x_tipTE = wing_span / 2
    y_tipTE = - wing_span / 2 * np.tan(sweep) - chord_tip

    sequence_X = [x_fuselageLE, x_rootLE, x_kinkLE, x_tipLE, x_tipTE, x_kinkTE, x_rootTE, x_fuselageTE,
                  -x_rootTE, -x_kinkTE, -x_tipTE, -x_tipLE, -x_kinkLE, -x_rootLE, x_fuselageLE]
    sequence_Y = [y_fuselageLE, y_rootLE, y_kinkLE, y_tipLE, y_tipTE, y_kinkTE, y_rootTE, y_fuselageTE,
                  y_rootTE, y_kinkTE, y_tipTE, y_tipLE, y_kinkLE, y_rootLE, y_fuselageLE]

    return sequence_X, sequence_Y


if __name__ == '__main__':
    # Select the analysis to perform
    ANALYSIS = [0,  # Basic Single Wing Evaluation
                0,  # General performance
                0,  # Plotting optimized wing planforms
                0,  # Design Vector variable analysis
                0,  # Get Failure rate of ML
                0,  # Tensorboard read results ,Performance vs trained timesteps | Training table to graph
                0,  # Rewards generated vs objective value
                0,  # Optimization progression vs timestep
                0,  # Constraints progress vs timesteps
                0,  # model resilience
                0,  # General performance of No retraining vs clean retraining vs continual learning vs transfer learning
                1,  # planforms TL DA50
                0,  # design variable changes(max), average, F50,ASK21
                0,# rewards vs objective
                0, #DA50 planforms progressive , SR22 planforms progressive
                0, #F50 planforms
                0,#ASK21 planforms
                0,# Average higher objectives
                0,#tensorboard plot
                0] #Sigma Analysis

    ########### GENERAL PERFORMANCE ####################

    #### ANALYSIS 1: Basic Single Wing Evaluation
    # Constraints check
    # Get final values
    # print outs
    if ANALYSIS[0] == 1:
        aircraft_config = AC.aircraft_config

        wing_span = AC.wing_span_start
        chord_root = AC.chord_root_start
        chord_kink = AC.chord_kink_start
        chord_tip = AC.chord_tip_start
        sweep_le = AC.sweep_le_start
        yoffset_kink = AC.yoffset_kink_start  # at 50% includes fuselage width
        zoffset_tip = AC.zoffset_tip_start
        twist = AC.twist_start
        thickness_chord = AC.thickness_chord_start

        # get design vectors
        design_vector_start = [wing_span, chord_root, chord_kink, chord_tip, sweep_le, yoffset_kink, zoffset_tip, twist,
                               thickness_chord]

        design_vector = [1.497273867207e+01, 1.568192553962e+00, 1.445681755345e+00, 5.320311550767e-01,
                         0.000000000000e+00, 3.379172863454e+00, 1.851584110870e-01, 3.677958950478e-01,
                         1.388268189363e-01]
        design_vector = [15., 1.14127798, 1.14127798, 1.14127798, 0.20849882, 2.43142124,
                     0., -1.19255368, 0.13904111]
        # design_vector = [15.    ,      1.22769383 , 1.22769383 , 1.22769383 , 0. ,         2.21354333,  0.   ,      -1.61572937 , 0.14116904]

        objective_return_start = objective(design_vector_start)

        print(objective_return_start)

    #### ANALYSIS 2: General Performance
    # obtain Max objective, average, standard deviation and average time per optim
    if ANALYSIS[1] == 1:
        # Get printouts
        # PSO results from folder extractor
        aircraft_config = 'SR22'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        print(CL_CD_PSO)
        print('max ', max(CL_CD_PSO))
        print('ave', sum(CL_CD_PSO) / len(CL_CD_PSO))
        print('stnd', np.std(CL_CD_PSO))
        print(design_vectors_PSO)

    #### ANALYSIS 3: Plotting optimized wing planforms
    # Process X best wings from total list
    # Plot X planforms with twist, dihedral and thickness in legend
    if ANALYSIS[2] == 1:  # ML planforms

        design_vector_start = [AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start,
                               AC.sweep_le_start, AC.yoffset_kink_start, AC.zoffset_tip_start,
                               AC.twist_start, AC.thickness_chord_start]

        # SORTING ML RESULTS
        number_ML = 100
        CL_CD_ML = res.objectives
        design_vectors = res.design_vectors
        sorted_CL_CD_ML = sorted(CL_CD_ML)
        best_CL_CD_ML = sorted_CL_CD_ML[-number_ML:]
        best_CL_CD_ML.reverse()
        best_idx_ML = []
        best_design_ML = []

        for idx in range(len(best_CL_CD_ML)):
            best_idx_ML.append(CL_CD_ML.index(best_CL_CD_ML[idx]))

        for idx in range(len(best_CL_CD_ML)):
            best_design_ML.append(design_vectors[best_idx_ML[idx]])

        # PLOTTING planforms results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        plt.axis('equal')
        # plt.title(f'Optimized SR22 wing planforms')
        x_AC1, y_AC1 = points(design_vector_start, AC.fuselage_width)
        label1 = 'Reference semi-planform'
        plt.plot(x_AC1[:8], y_AC1[:8], color='#969696', label=label1)
        plt.vlines(0, -2, 0.8, colors='#525252',linestyles='dashed')




        plt.plot()
        x_AC_ML = []
        y_AC_ML = []
        for design in range(number_ML):
            x_ACi, y_ACi = points(best_design_ML[design], AC.fuselage_width)
            x_AC_ML.append(x_ACi)
            y_AC_ML.append(y_ACi)
            # print(x_AC_ML)
            label_ML = f'PPO optimized semi-wing'
            plt.plot(x_AC_ML[design][:8], y_AC_ML[design][:8], color='#74a9cf')

        label_ML = f'PPO optimized semi-planform'
        plt.plot(0, 0, color='#74a9cf', label=label_ML)
        label_PSO = f'PSO optimized semi-planform'
        plt.plot(0, 0, color='#fd8d3c', label=label_PSO)
        #####PSO PLOTTING
        number_PSO = 1
        aircraft_config = 'SR22'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        sorted_CL_CD_PSO = sorted(CL_CD_PSO)
        best_CL_CD_PSO = sorted_CL_CD_PSO[-10:]
        best_CL_CD_PSO.reverse()
        best_idx_PSO = []
        best_design_PSO = []

        for idx in range(len(best_CL_CD_PSO)):
            best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

        for idx in range(len(best_CL_CD_PSO)):
            best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

        # print('ML DV', best_design_ML[0])
        # print('PSO DV', best_design_PSO[0])
        # PLOTTING ML results
        x_AC_PSO = []
        y_AC_PSO = []

        for design in range(number_PSO):
            x_ACi, y_ACi = points(best_design_PSO[design], AC.fuselage_width)
            x_AC_PSO.append(x_ACi)
            y_AC_PSO.append(y_ACi)
            # labeli = f'PSO design {design + 1}, CL/CD:{round(best_CL_CD_ML[design],3)}'
            plt.plot(x_AC_PSO[design][:8], y_AC_PSO[design][:8], color='#fd8d3c')

        fig.gca().legend(loc="upper left")
        # fig.axes.yaxis.set_ticklabels([])
        plt.show()

        # Extract ML results.
    if ANALYSIS[2] == 2:  # PSO planforms

        number_of_best = 10
        design_vector_start = [AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start,
                               AC.sweep_le_start, AC.yoffset_kink_start, AC.zoffset_tip_start,
                               AC.twist_start, AC.thickness_chord_start]
        # get directory
        aircraft_config = 'SR22'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        sorted_CL_CD_PSO = sorted(CL_CD_PSO)
        best_CL_CD_PSO = sorted_CL_CD_PSO[-10:]
        best_CL_CD_PSO.reverse()
        best_idx_PSO = []
        best_design_PSO = []

        for idx in range(len(best_CL_CD_PSO)):
            best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

        for idx in range(len(best_CL_CD_PSO)):
            best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

        # PLOTTING ML results
        fig = plt.figure()
        plt.axis('equal')
        plt.title(f'RL optimized wing of {AC.aircraft_config} wings at cruise')

        x_AC1, y_AC1 = points(design_vector_start, AC.fuselage_width)
        label_PSO = 'PSO optimized wings'
        plt.plot(x_AC1, y_AC1, color='blue', label=label_PSO)

        x_AC = []
        y_AC = []

        for design in range(number_of_best):
            x_ACi, y_ACi = points(best_design_PSO[design], AC.fuselage_width)
            x_AC.append(x_ACi)
            y_AC.append(y_ACi)
            # labeli = f'PSO design {design + 1}, CL/CD:{round(best_CL_CD_ML[design],3)}'
            plt.plot(x_AC[design], y_AC[design], color='red')

        fig.gca().legend(loc="upper left")
        plt.show()

    #### ANALYSIS 4: Design Vector variable analysis
    # get all design vectors of optimized wings
    # Make boxplot of each variable
    # plot average change shape to compare
    if ANALYSIS[3] == 1:
        objectives = res.objectives
        design_vectors = res.design_vectors
        collected_design_variables = []
        normalized_design_variables = []

        # get directory
        aircraft_config = 'SR22'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        design_vectors = design_vectors

        for dv in range(len(design_vectors[0])):
            collected_vars = []
            for i in range(len(design_vectors)):
                collected_vars.append(design_vectors[i][dv])
            collected_design_variables.append(collected_vars)

        for dv in range(len(design_vectors[0])):
            normalized_vars = []
            for i in range(len(design_vectors)):
                norm_var = (design_vectors[i][dv] - AC.lower_bound[dv]) / (AC.upper_bound[dv] - AC.lower_bound[dv])
                normalized_vars.append(norm_var)

            normalized_design_variables.append(normalized_vars)

        # meanpointprops = dict(marker='D', markeredgecolor='black',
        #                       markerfacecolor='firebrick')

        fig = plt.figure(figsize=(10, 7))
        ax1 = fig.add_subplot(111)
        plt.title(f'PPO')
        # Creating axes instance
        # ax = fig.add_axes([0, 0, 1, 1])
        labels = [r'$b$', r'$c_{root}$', r'$c_{kink}$', r'$c_{tip}$', r'$\Lambda_{LE}$', r'$y_{kink}$', r'$z_{tip}$', r'$\epsilon$', r'$t/c$']
        # Creating plot
        bp = ax1.boxplot(normalized_design_variables, labels=labels)
        # x = np.linspace(1,9,9)
        # ax1.xticks(x, labels, rotation='vertical')



        design_vector_start = (
        AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start, AC.sweep_le_start,
        AC.yoffset_kink_start, AC.zoffset_tip_start, AC.twist_start, AC.thickness_chord_start)
        for var in range(len(design_vector_start)):
            plt.hlines((design_vector_start[var] - AC.lower_bound[var]) / (AC.upper_bound[var] - AC.lower_bound[var]),
                       0.6 + 1 * var, 1.4 + 1 * var, colors='#238b45')
        plt.plot(0.6,(design_vector_start[0] - AC.lower_bound[0]) / (AC.upper_bound[0] - AC.lower_bound[0]),color='#238b45' ,label='Starting value')
        fig.gca().legend(loc="upper right")
        plt.hlines(0, 0.75, 9.25, '#525252')
        plt.hlines(1, 0.75, 9.25, '#525252')

        x = np.linspace(1, 9, 9)
        for var in range(len(design_vector_start)):
            plt.vlines( 0.5+ 1 * var, 0, 1, colors='#bdbdbd', linestyles='dashed')
        plt.vlines


        # ax.set_xticks(x,color = '#525252', labels, rotation='horizontal')


        # show plot
        plt.show()

    #### ANALYSIS 5: Get Failure rate of ML
    # Get PSO average
    # Put objective values in bins
    # Simple bar graph compared to the PSO average
    if ANALYSIS[4] == 1:
        objectives = res.objectives
        PSO_CLCD, PSO_design_vectors = get_PSO()
        PSO_average = sum(PSO_CLCD) / len(PSO_CLCD)
        bins = np.linspace(28, 31.2, 34)
        x_scale = []
        PPO_average = sum(objectives) / len(objectives)
        for idx in range(len(bins)):
            cntr = 0
            for ob in objectives:
                if ob > bins[idx] and ob < bins[idx + 1]:
                    cntr += 1
            x_scale.append(cntr)
        print(bins)
        print(x_scale)

        # PLOTTING

        fig = plt.figure(figsize=[7, 5])
        ax = plt.subplot(111)
        # ax.set_xticks(bins)

        # set the basic properties
        # ax.set_xlabel('x axis')
        # ax.set_ylabel('y axis')
        # ax.set_title('title ')
        # xlab = ax.xaxis.get_label()
        # ylab = ax.yaxis.get_label()
        #
        # xlab.set_style('italic')
        # xlab.set_size(10)
        # ylab.set_style('italic')
        # ylab.set_size(10)

        # set the limits
        # ax.set_xlim(0, 24)
        # ax.set_ylim(6, 24)

        # l = ax.fill_between(xdata, ydata)
        ax.grid('on', linestyle='dashed')
        plt.hist(objectives,200, color='skyblue', label= 'PPO result distribution')
        plt.vlines(PPO_average, 0, 30, colors='#08519c', label='PPO average result')
        plt.vlines(PSO_average, 0, 30, colors='#feb24c',label = 'PSO average result')

        plt.xlabel('Objective value')
        plt.ylabel('n')
        fig.gca().legend(loc="upper left")
        plt.show()

        ########### TRAINING ####################
    #### ANALYSIS X:
    #### ANALYSIS 6: General training process
    # Tensorboard graph of average reward and timestep

    if ANALYSIS[5] == 1:
        x=3




        ########### TRAINING ####################
    #### ANALYSIS 7: Performance vs trained timesteps

    if ANALYSIS[5] == 2:
        max_obj = [30.794,30.728,30.877,30.859,31.071 ,31.094,31.164,31.173,31.162]
        ave_obj = np.array([28.771,29.529,30.422,30.473,30.695,30.742,30.961,30.995,30.991])
        std_1 = np.array([0.693, 0.755, 0.455, 0.370, 0.293,0.341,0.254,0.190,0.134])
        PSO_ave = 30.84
        start_obj = 28.08
        time_steps = [10000 , 26000,50000 , 100000,250000,500000,1000000,1272000,1500000]
        training_time = [0.257, 0.6833, 1.3,2.6833, 6.62, 13.35, 33, 40,47]



        fig,ax1 = plt.subplots()
        color = '#0570b0'

        # plt.plot(time_steps, ave_obj, color='#74c476', label='Average objective')
        plt.xlabel('Training steps')
        plt.ylabel('Reward value')

        ax1.set_xlabel('Training steps')
        ax1.set_ylabel('Objective value', color=color)
        # ax1.plot(t, data1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.plot( time_steps[0],30.794,color='#238b45', label = 'Training time' )
        ax1.plot(time_steps, max_obj, color='#045a8d', label='PPO maximum' )
        ax1.plot(time_steps, ave_obj, label='PPO average',color = '#3690c0')
        ax1.fill_between(time_steps, ave_obj - std_1, ave_obj + std_1, color='#67a9cf', alpha=0.2)

        ax1.hlines(PSO_ave, 0, time_steps[-1], color='#fd8d3c', label='PSO average',
                   linestyles='dashed')
        ax1.fill_between(time_steps, PSO_ave - 0.031 ,  PSO_ave + 0.031, color='#ef6548', alpha=0.2)
        ax1.grid('on', linestyle='dashed')
        fig.gca().legend(loc="lower right")
        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        # plt.hlines(start_obj,0, time_steps[-1],  color='#737373', label='Starting Objective', linestyles='dashed')

        color = '#238b45'
        ax2.set_ylabel('Training time [hr]', color=color)  # we already handled the x-label with ax1
        ax2.plot(time_steps, training_time, color=color, label = 'Training time')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.legend()

        # plt.show()

        plt.show()


        # plt.plot(step, CL_CD_list, color='orange', label='Objective')



        # plt.title(f'Rewards vs step')


    # Results needed: Performance table at different training steps.
    #### ANALYSIS 8: Rewards generated vs objective value vs trained timestep
    # Multiple graphs at progressive timesteps to correlate the rewards obtained vs performance

    if ANALYSIS[6] == 1:
        aircraft_config = AC.aircraft_config
        # Res = Results('SR22_1_272_000')
        CL_CD_list = res.objectives
        total_rewards = res.total_rewards
        penalties = res.penalties
        rewards = res.rewards


        # Get a fitted curve for total_rewards progression
        def fit_curve(x, a, b, c):
            return a * x + b * x ** 2 + c


        popt, _ = sp.optimize.curve_fit(fit_curve, CL_CD_list, total_rewards)
        a, b, c = popt
        x_line = np.arange(min(CL_CD_list), max(CL_CD_list), 0.1)
        y_line = fit_curve(x_line, a, b, c)
        y_line = fit_curve(x_line, a, b, c)

        # Plotting of the rewards progression vs CL_CD
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.scatter(CL_CD_list, total_rewards, s= 3,color='#4292c6', label='Total rewards')
        plt.scatter(CL_CD_list, penalties, s= 3,color='#ef6548', label='Penalties')
        plt.scatter(CL_CD_list, rewards, s= 3,color="#78c679", label='Gains')
        plt.plot(x_line, y_line, color="#045a8d", label='Fitted total rewards')
        plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color="#737373", linestyles='dashed')
        plt.xlabel('Objective value')
        plt.ylabel('Reward value')
        fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs Objective reached for {aircraft_config}, 500 000 training steps')
        plt.show()
    if ANALYSIS[6] == 2:
        aircraft_config = AC.aircraft_config
        # Res = Results('SR22_1_272_000')
        CL_CD_list = [30.623709253512928, 30.286695717456855, 30.335391610188665, 30.712867613912536, 30.78384957960856, 30.37973586206115, 30.644541603915243, 30.647965732960383, 30.352123778537038, 30.202935746416234, 30.441461741125202, 30.678229879005126, 30.323646549560884, 30.575487003994308, 30.610144994305138, 30.859486503273576, 30.564516755583593, 30.63471466861933, 30.80219568390154, 30.346665802114863, 30.628977808655044, 30.486572050872553, 30.709708497174105, 30.62828532983853, 30.591386158297368, 30.620481059458516, 30.47836181802909, 30.707439070947466, 30.729587888002573, 30.4105026522973, 30.59672228265973, 28.599734818712662, 30.723747977866115, 30.422018504670937, 30.557890141401124, 30.61102411159956, 30.50947189513561, 30.362426670136234, 30.685494826835246, 30.614774964046767, 30.705335043204744, 30.533829727111527, 30.468424533790824, 30.437550283723212, 30.665201086909878, 29.64694565081532, 30.682321158730986, 30.305869336842225, 30.460745754700113, 30.635166381531246, 30.529574834136877, 28.937667972176165, 30.67349588455209, 30.272796509943703, 30.78838679902113, 30.74386035615184, 30.56843934477686, 30.25174905774852, 30.601047568767996, 30.64320241391871, 30.5885735926055, 30.241741019789686, 30.664008524415273, 29.09946261296599, 30.47897944652923, 29.666527715588384, 30.481387846315943, 28.374655206307537, 30.517830223894023, 30.708220915226345, 30.791021478091377, 30.573986324531305, 30.271588885121268, 30.697244912520617, 30.690756921509614, 30.61292983866998, 30.364084999550357, 30.49546050843928, 30.68792021650687, 30.471675087081426, 30.44298484847458, 29.770619927976234, 30.795869962453256, 30.498568586773384, 30.645595294818825, 30.434853302086672, 30.38509947623601, 30.55795153897926, 30.61448369598876, 30.485722711261413, 30.51860602741975, 29.922224235346597, 30.596259731238806, 29.001866693255344, 30.699922724989005, 30.71327793359239, 30.755584890678335, 30.70382090784282, 30.162146331310915, 30.584614085397785, 30.702205205648, 30.64470783520291, 30.497984691682163, 30.55601472114476, 30.4362695092311, 30.58387391383226, 30.50638237982434, 30.63104803092039, 30.711878181896473, 30.083340167038244, 30.623168519967457, 30.51492237902337, 30.45696661612082, 30.565609548286755, 30.282658697323605, 30.26858218999391, 29.837178194989807, 30.681291456390028, 30.72512441808087, 30.36718688528158, 30.488261475863197, 30.365867131996247, 30.74396017883488, 29.025610973898864, 30.58424890973745, 30.781568981183543, 30.541115916088433, 30.64759582051086, 29.977875210534762, 30.720035118175435, 30.49622246577206, 30.500853311962867, 30.74573358514442, 30.575891567528835, 30.508228391516823, 30.65698161868743, 30.523336551301128, 30.613441354468602, 30.75281621296813, 30.47821836645745, 30.635024170667098, 30.71412830060748, 30.725131629579664, 30.47694321450239, 30.361820610100786, 30.436913392210226, 30.568638944141355, 30.505761731501746, 30.388301080443842, 30.687773993721667, 28.08, 30.65000062457458, 30.539395071876413, 30.669271168552605, 30.678910638196612, 29.805774693301682, 30.41598265598678, 30.6377097920934, 30.71506705743986, 30.625975065865266, 30.18787453527077, 29.94802763935992, 28.08, 30.576600107368073, 29.91454268696464, 30.692027539684496, 30.59588963005017, 30.542519287887604, 30.350010972663693, 30.332023987052498, 29.57692608984022, 30.641832942126612, 30.503352656035585, 30.610426219903708, 30.492482452236104, 30.76163913934855, 30.41603331872485, 30.64420842105538, 30.57427212088119, 30.56609797313678, 30.495612368999737, 30.32016961390036, 30.654089834841862, 30.48301191720476, 30.482223135079938, 30.50715364488497, 30.66369932178199, 30.436887955052626, 30.748232677859203, 30.480493458786654, 30.628903202755595, 30.76328443197687, 30.699708448156798, 30.69305644569705, 30.568508625930644, 30.633218827683326, 30.481645837342228, 30.684098049719545, 30.603934620678956, 30.469130872152903, 30.562846408766834, 30.4313498734059, 30.635571840969074, 30.65368727659787, 30.75373793563414, 30.377499185294134, 30.812855029832537, 30.391913026678523, 30.702033248602717, 30.3879673808927, 30.751016948394774, 30.339856784521672, 30.640767728720004, 30.50181771934958, 30.66003796833949, 30.35201384414984, 30.517530392051444, 30.3179455924899, 30.403349376688002, 30.431576726367084, 30.67553333742082, 30.41094706923478, 29.092318011640483, 30.66673330417605, 30.190654686567445, 30.633692406518833, 30.65915853702716, 30.57270525035405, 30.60132443983963, 30.728273939156274, 30.620682392554873, 30.65426260193713, 30.741477235404176, 30.490992574475307, 30.53539912892225, 30.540845635641475, 30.460455803579052, 30.60971717314822, 30.688485204786698, 30.59720440414534, 30.555212708204166, 30.274460067611425, 30.678683476541067, 30.589517704887268, 30.777927263337617, 30.716470797790667, 30.604934793415296, 30.493771993189256, 30.744705044750884, 30.671882406287683, 29.506872492051137, 29.897326783231744, 30.53392424030969, 30.75179888520689, 30.671604481715942, 30.54216872815862, 30.80086286917064, 30.310760552286318, 30.5454214457522, 30.604119175024184, 30.238019230020715, 30.630761645186517, 30.64854169388475, 30.522956151900818, 30.529421253539603, 30.747874571400722, 30.70270275777259, 30.59321874186939, 30.706100455690557, 30.703529180271786, 30.689027689586958, 30.549206101908222, 30.830499672470065, 30.640961317830577, 30.787218123309994, 30.68095126705139, 30.26376067431495, 30.748532357694195, 30.506016803765462, 30.492191533129656, 30.424507101004608, 30.56275370219619, 30.353717414111568, 30.35192880886605, 30.610361183400595, 30.48424863105833, 30.609798665398923, 30.490563961493077, 30.7615678271734, 30.280769831595453, 30.78791357755369, 30.532089833142308, 30.704316460627414, 30.48337458013028, 30.416715426777316, 30.486603211159103, 30.585443614409428, 30.4835129772324, 30.6163671405417, 30.74359481859062, 30.66412196381735, 30.743740574009838, 30.610732079581474, 30.59014825336836, 30.699625617059176, 30.450019748130753, 30.42850613072883, 30.53522810043787, 30.657750809089862, 30.39633877317088, 30.383534079584862, 30.482105048803543, 30.6296924302192, 30.6922394801151, 30.495698288749516, 30.528302385063956, 30.603466269220554, 30.046776045098678, 30.365721824334152, 30.511662128466277, 29.53676273967017, 30.40384131458533, 30.490443681701276, 30.388184418441085, 30.479626204045356, 30.689958438814934, 30.025000551943226, 30.454865486676272, 29.948266062095644, 30.65200414602466, 30.625085124863922, 30.56052658454818, 30.74617739760341, 30.70686583046446, 30.52410853011638, 30.685853782885626, 30.466707751499296, 30.633717692308966, 30.683684102520843, 30.186593127882418, 30.242683132514482, 29.983393447134247, 30.18379186318336, 30.668206867886937, 30.814003284801093, 30.355826457331535, 30.691736669906867, 30.575310648328916, 30.63269687945501, 30.768416495736552, 30.56713317824458, 30.721741077896155, 30.42360171676278, 30.342169820453346, 30.01007340010151, 28.83663962243798, 30.408517823782677, 30.710618034890448, 30.728531160961982, 30.731208066213604, 30.65106095032164, 30.106741656602, 29.560937571616023, 29.51727029913054, 30.521554458017732, 30.56838221111725, 30.651913474709485, 30.610997840299163, 30.369310851882634, 30.61451698824464, 30.65111558383004, 28.08, 30.35400300419183, 30.598560295258405, 30.722743847747275, 30.582663239105955, 30.466848188046267, 30.426551172979607, 30.43049492828157, 30.587912730590062, 30.451211905189375, 30.682373563915235, 30.64517645505709, 30.461641693615274, 30.70438395175938, 30.724058717240105, 30.550653919055154, 30.62946811797443, 30.424109945403146, 30.46362195592701, 30.614646806503725, 30.585616721693178, 30.50761518822514, 30.618827192886986, 30.313483023432948, 30.62376499780637, 30.44214657307585, 30.613299379771757, 30.742130366984412, 30.659444415961378, 30.57283823209574, 30.772994172896045, 30.595446614726715, 30.57226867846806, 30.35054895491742, 30.58747773963793, 30.475767029547185, 30.633104488139207, 30.498081946233686, 30.608058828661562, 30.033806244554548, 30.58768494033211, 30.560600492782275, 30.47775618681226, 30.544001278870685, 30.769005254436728, 29.96457812119177, 30.43890282057384, 30.734526661568353, 30.404508071607392, 29.527051859052534, 30.70866879271271, 30.531743512310186, 30.6277990164666, 30.483343897070256, 30.730229247785342, 30.520131534745214, 30.498966051525198, 30.71315969675125, 30.33115624608805, 30.44856667470082, 30.628429193645548, 30.441617720199805, 30.65082494880916, 29.975068303715343, 30.033766418672162, 30.452700123352646, 30.548817754085423, 30.513050307669126, 30.618666525782565, 30.702792450909087, 30.60704927665592, 30.4471204070064, 30.39044184492032, 30.356608853752455, 30.689650765015973, 30.25591496970991, 30.2888342081847, 30.449989684769193, 30.44210606877683, 30.52805780493212, 30.579582519013055, 30.65851016284594, 30.700544331335745, 30.597370277715324, 30.724429846886007, 29.09995089229708, 30.48026883597504, 30.65409244657627, 28.952755001744414, 30.46203817529743, 30.68364114441477, 30.46127207701203, 30.714197811333992, 30.366925124026825, 30.467075601525433, 30.69804341036924, 30.313371886386626, 30.83907022906123, 30.022039054740663, 30.047183877435437, 30.499278443665307, 30.673591198855735, 30.739265801865784, 30.69723946170922, 30.813967324069665, 30.749724147366713, 30.746933988324546, 30.51887452284464, 30.153985853335747, 30.490277783345938, 30.57585480529356, 30.601145516128277, 30.64710052629253, 30.634212902244478, 30.698106302341948, 30.28545701589227, 30.757836712792763, 30.702897094989076, 30.703660533770805, 30.660863965135558, 30.15434985615307, 30.520142234783243, 30.780031730593898, 30.413680960617167, 30.736747338419704, 29.55199574983128, 30.62674059361576, 30.297631471989256, 30.683579130198897]
        total_rewards = [312.96907417531384, 116.79852361482614, 179.57223960642202, 311.2196059130007, 380.28545940234164, 150.67397677910125, 248.7893047107251, 368.90567210660697, 131.9435394818131, 164.71550173322237, 223.38614767448973, 237.18995153620838, 173.6301042749621, 378.49590443778555, 206.92951156411772, 297.5927287961368, 244.37507041940998, 224.81077399076668, 447.7910333390005, 86.81248144374831, 244.0057647246648, 172.44665174435318, 215.5049462185384, 291.19084112772555, 268.55516315514683, 267.50558748121625, 211.2650906486897, 228.66768810636736, 207.69024839405515, 178.74952634534213, 211.2979557064932, -27.00700637493061, 308.3313875453734, 228.96354705148434, 154.87570129251634, 282.6206435875931, 143.9573391375676, 120.87225654491462, 334.46358983105864, 271.3308736950647, 341.7747038223607, 177.74866948210556, 213.31197287580864, 289.3017909176806, 249.82478283646861, 12.410256586916528, 297.6602606795109, 179.86445861366667, 212.04445146084157, 274.4039110106422, 219.15238881413177, -50.14956381084927, 262.2410543331864, 159.44296735970485, 321.21898421022996, 297.33050416608774, 240.30898472652754, 195.5727614205462, 323.4152132629811, 162.01719341901526, 227.0235966413587, 149.0406866829368, 226.33524959622878, -22.503032267458273, 195.7982158720218, 51.254241114900275, 238.77174802635562, -34.890781726995066, 271.4442237346326, 277.97715311988713, 375.29518216404097, 191.96729415509816, 197.70578749886812, 326.19800072039345, 325.28979944367063, 225.96957594082266, 135.54795036179036, 233.86499044402615, 281.06965405217534, 273.1511691250158, 190.1976106311023, 71.56967335343245, 319.1734540760778, 249.6905531919785, 262.1872297281829, 157.99432047260925, 164.60986028861075, 232.20419222674104, 266.5692408974801, 246.8760982756668, 176.0167350605572, 64.12307825875313, 195.11869370025053, -28.787139377574338, 231.41802937945675, 310.68350245899586, 368.65533233693907, 333.04599718509246, 116.84386330840087, 189.03214308841194, 365.0958944541581, 218.50110759533487, 165.218714047736, 211.86437947751756, 254.4595715594017, 183.9351187256027, 226.4551094295624, 252.666685425776, 316.2350072114932, 96.1171320797906, 283.0356438702346, 112.98801558562415, 171.822937476574, 258.0669923077412, 224.9990941977227, 150.66544872795663, 19.413082615760338, 307.95889468569965, 277.2480864902438, 139.53674929821076, 231.69027540168992, 176.1393976787982, 231.2865977376363, -24.695132257889796, 178.46519141821028, 348.5600637956071, 245.08486957586686, 305.39785685562913, 74.20924222577736, 255.55994375954702, 261.10084918179996, 276.29332117970824, 226.9813703985659, 74.76998427080444, 320.7718901329071, 164.72836090534028, 197.3263729129651, 233.92992976202203, 330.7572338144935, 132.3478415418013, 333.2594098715525, 294.4989135264552, 274.17472408845964, 203.3455796348591, 155.64037380432455, 204.2171571639829, 280.74894470929286, 153.46228722100315, 211.18681832860204, 245.23597326869725, -42.24377888990478, 307.64628727975935, 237.3037688144493, 252.89358024157968, 295.599619943312, 29.298533466939997, 136.69164746992874, 242.9378345578084, 241.31849373761952, 292.95410643100456, 179.22028175295202, 3.376058405362045, -36.55597084354665, 283.80459313364224, 148.22931861306176, 209.30362283623447, 267.7015003414493, 178.7829909124552, 142.0635868902679, 170.8513918024589, 0.6567230159571835, 148.18303742940878, 199.8991362767718, 295.14414754105206, 255.55032440001207, 364.5671504324226, 197.4781764222371, 273.51640295408157, 310.082067572183, 328.5048784472406, 291.41970559147495, 190.94020860410242, 278.18257553314186, 173.0489851912128, 243.38011082238458, 223.78977076533204, 178.24737752371726, 240.72881816696025, 390.16290118679626, 275.82150468292093, 234.53472079808688, 346.621852692145, 285.689811063084, 298.98219357557997, 106.82771460253039, 238.25857334554274, 154.66153442005879, 322.3728209751927, 271.6365766289589, 282.9957427450084, 195.2914177374971, 143.70890753619457, 282.1559516867952, 333.82523743790466, 337.6034677675591, 207.2035705479942, 334.86401392827605, 150.99887737171431, 214.36437113733547, 139.05722369777288, 241.43280173548214, 261.40928988799163, 347.22381542450125, 328.4952454420884, 240.91913396679882, 191.73016458164992, 160.0895201155986, 196.308397807225, 224.2781618644488, 253.91592448707894, 303.47060077936857, 213.51946615347967, 4.627850009789981, 255.90800983172602, 154.87414929096747, 205.25974123525697, 248.04727351366625, 216.15529866743395, 246.61371375718716, 296.0365426027461, 246.38546953809043, 267.7456322315587, 252.7683424695453, 238.55334619478333, 344.22359263692596, 158.8892569762195, 235.0309625076496, 349.8765656894094, 290.1583522859108, 364.8868784255169, 340.30244980209886, 162.11094216688258, 346.9511919957765, 291.27498804893247, 285.08539997819196, 228.46414932414115, 276.9104460217389, 208.60688676527013, 305.01267082637287, 223.61184441235775, 33.96137959394947, 80.09465237055875, 227.88851447745918, 420.8250165345356, 312.7546687150055, 150.39710236679258, 346.7390101122853, 253.64409814953083, 346.3016644642842, 226.73534795226715, 127.44214438834021, 243.55529566284287, 206.96176215427067, 132.47408477352766, 205.70562227759834, 438.71758047019364, 246.6610774163512, 402.0100285330702, 290.8235802677137, 442.77348320639544, 227.14970032435224, 241.59363218202844, 310.07562249547766, 172.3437271687175, 327.2984188968974, 251.16097766034386, 140.08811812751227, 177.21246713133687, 223.62303840168906, 228.5354401139793, 250.09968883398432, 180.63709647453908, 148.60178798845993, 128.2149225875167, 169.51321480691223, 131.07930385451962, 301.80188585283184, 209.2723551404781, 259.65249934043027, 125.48346492968531, 351.73607963068196, 273.3587481655003, 320.77088320306564, 248.59772944338954, 131.5426459859906, 126.71734016139334, 267.16054937021806, 240.0039918220793, 325.2162658924925, 450.5979557347419, 319.6609339132817, 372.6547081463058, 330.34568437646965, 254.88211733545137, 189.4739202704164, 232.53068571121463, 209.3844313682194, 195.1153046500733, 259.8485367843151, 183.1944084295472, 235.87592738605298, 173.98548748178152, 294.1377366748154, 311.77317870578236, 213.02459178862105, 194.20167142212745, 146.55509402452745, 137.87827747527606, 188.12740953412666, 225.8342155926041, 39.916265718737996, 196.92965483333614, 155.79132488210752, 184.03232192692872, 157.12218150846792, 233.3078737962672, 56.503555223238166, 189.71585105952806, 3.366420033804438, 255.0774410744935, 269.5090123064482, 296.32465895259963, 322.53256520459183, 363.779576521373, 190.8897216971082, 280.53104688411827, 283.9601624703297, 221.52596487778135, 326.7989764189109, 196.74995258304034, 173.90051904402242, 81.47201008357611, 129.47572719816435, 262.6517133059848, 381.80820101427247, 194.44774508505313, 239.84492756866257, 281.72637194227605, 335.89308158849246, 341.72263091089025, 339.5298726625881, 293.4497143718539, 222.14893647734988, 142.099002756513, 157.27396740517713, -27.22239676312099, 200.67819402280392, 350.70983982621476, 344.1929042877214, 243.72931722104852, 207.39034215087213, 17.500341202327135, 48.28285046746884, -16.13019413158476, 232.64872578967976, 231.04497620957432, 338.27345178556084, 251.41094488206286, 119.95921755377184, 237.90385010822183, 145.3156097355971, -32.77769284394237, 130.4740487247381, 194.1962705616692, 313.0424440050251, 219.79434986679473, 223.15874300377806, 196.52272144316078, 256.03403535486393, 146.51879275566873, 210.1980905745035, 292.69005775403195, 210.16275353699118, 291.4662078056855, 235.27070932509295, 387.06994496522447, 219.8786066970247, 278.30125075657605, 188.1034842418942, 314.59108396011567, 213.2900692422411, 208.65984531339765, 239.6048290141556, 151.33357243785062, 215.2076292660587, 275.5703013044085, 235.52722452833075, 279.16526303822576, 276.14285728122417, 198.50894234905593, 271.86290890142425, 306.2417447809781, 270.0657871648421, 162.85894126756074, 148.09672410237445, 164.53787349024287, 294.71388954050286, 306.5694657626457, 285.59190350261355, 353.70567990728125, 104.93905562819627, 266.8473242199441, 299.3162430129464, 253.7628952080127, 263.0856385610518, 292.6324639532355, 120.53531607672258, 224.37629509268197, 348.5726125535018, 262.5302740389318, -2.2879330753020213, 283.25327947791664, 257.77054566717175, 324.31971679902904, 181.92696988088738, 328.57265197171125, 268.114509986405, 265.01134693167455, 227.34041972812514, 115.37268420081227, 126.10971672250378, 227.90860151127652, 114.64485505051297, 232.697724748994, 73.71948046696824, 123.50069890093653, 227.09448016940507, 231.65791405131287, 310.3739537059296, 231.08564702624432, 281.94777847712487, 300.2139879578848, 227.60224582255552, 73.26652861897941, 157.70420699704513, 290.66895493742635, 131.56578246902785, 170.2376774889463, 165.87436256291008, 223.117083468307, 154.89594472057627, 242.06529131517692, 378.5024123042185, 314.3994867323633, 288.5457068262683, 301.3600481994723, -24.06691037329686, 141.13669192295725, 222.7028943296301, -19.336550997947903, 291.6899290462074, 243.32536411775072, 285.96659861143735, 268.4359206295108, 235.16227304529926, 171.76433340838045, 275.15057786293255, 155.6333424475007, 297.9438247738555, 99.41175686901295, 81.08963071677496, 236.58907453466404, 274.2189610636253, 378.6309356660389, 366.3485244068636, 328.17317641786025, 391.0918740615417, 376.4591974905709, 243.74894110597197, 107.55289083788074, 181.83518216561288, 265.3906884547948, 312.84335265093114, 328.6287006934042, 305.1538261142025, 331.7618083797378, 212.57627153279947, 280.4999449969761, 279.1888274842698, 290.60827368126206, 314.10576138718875, 187.69935009519784, 109.15355225254052, 261.0286469731931, 222.65094501177703, 319.1386727136759, 18.709887850851782, 264.8025568448536, 197.68897694852183, 317.43167440679554]
        penalties = [-62.30400743902356, -189.87082786201833, -86.21869468928756, -95.70713271841282, -64.87391930682007, -113.7664074538671, -93.56943775576352, -66.98674745641921, -215.4149447131046, -79.74537637538943, -107.94151807246207, -120.62192667722323, -105.492539928051, -73.06928306805548, -88.13758057764794, -95.6763676847645, -98.84423566643413, -91.9626074430318, -70.45686240047331, -86.02906844881433, -106.7505186633229, -87.11490322910527, -144.0874167641192, -87.54634348537972, -105.02794148007645, -72.81306629944423, -65.24165833628452, -103.24811552310722, -147.4062228328488, -147.7429157112298, -119.65346866648339, -39.29541614329321, -92.80353310122243, -76.943541300739, -84.30366672572038, -78.00489061872145, -205.1192645987182, -122.39362335855695, -93.52495249297061, -106.77853792009873, -85.35923219891068, -97.04155641586614, -98.46186528327357, -71.95290757310983, -65.83677644908101, -60.40043337777602, -55.66452984129058, -118.38794117993187, -83.53841070824232, -84.2541912171601, -63.14460889450041, -75.61595238377116, -82.26829180680105, -67.12467322142209, -70.81435400800832, -63.0994447942267, -72.68389835208478, -66.52468564482527, -67.82506177803596, -104.80740131525667, -79.16222776126048, -55.69955733036851, -83.21970418240403, -93.51940512062174, -113.57639059365383, -57.40424053870465, -78.71293496638914, -40.662226356837934, -112.67112876827188, -101.53691454974843, -62.01483575665215, -92.06275152833477, -84.2834984521451, -64.95781117616329, -95.63390426319668, -96.65985969291788, -89.8395729537746, -73.63877104703512, -78.00621042741435, -89.36632074306875, -85.79380404611152, -48.052962672271654, -130.71704705111458, -84.70923409337969, -61.343018587345156, -105.96751150753278, -162.1964090634444, -61.0369322645747, -121.09469817658557, -82.27639622719829, -85.6877362321332, -91.09254687949858, -88.60331582219698, -51.27532785556559, -121.5039509866954, -62.90468131970757, -79.09586771833479, -72.00175680686588, -67.8741145063142, -85.0805292336703, -66.52314637506342, -119.38572718105151, -85.84341039831205, -85.64259505121, -113.96276170597282, -106.5468303995893, -136.12567795760293, -77.44310478402382, -93.72384039576285, -126.75374559198451, -65.13258531887405, -130.0177397852326, -161.83867271027594, -76.57500243976125, -53.18534325648719, -89.23373130395254, -111.4932074168205, -77.05308776229622, -100.7865344862887, -78.223675272127, -91.2881065698834, -69.94068232475551, -104.13816100705408, -54.085320991456996, -107.20390274254986, -86.51983380584758, -64.45705690101585, -73.32280870239819, -101.712148745712, -68.48500307006313, -100.7534315036988, -93.33409695677568, -98.86221806920337, -158.99045397339074, -101.58763617826376, -124.39210511035353, -77.51771904800475, -99.79712865558349, -90.5978886813703, -119.00468292219352, -66.17377903481912, -98.9797771040101, -60.32765001784916, -74.17683545772152, -110.68596095351907, -86.97968395455473, -86.9365609233156, -102.34033578769282, -89.22225216383606, -107.81973986406226, -43.24377888990478, -78.87243173878869, -97.19870139014147, -111.74843086895866, -83.64950169575843, -141.5524103139752, -110.57642977515174, -62.94034163618868, -108.7330087080826, -94.58401776505363, -147.78534132835995, -92.44212310947199, -37.55597084354665, -59.96617076979452, -88.07810449369089, -158.741655651226, -67.77710560710784, -74.7080073614573, -139.62653175865842, -124.04752470255056, -50.02641713532124, -157.41511472258475, -122.26094872741206, -93.61329410683109, -87.15877687982612, -88.86667930683512, -78.056999836262, -81.50567721939566, -52.08195539703764, -60.90726875660414, -45.40857468825955, -54.84454963858982, -98.10656806482586, -77.17056707312102, -92.7513369080241, -103.73285456115923, -93.9662962333181, -92.8776523595536, -64.04346533428769, -117.55097761090705, -109.39613809918318, -62.08020694391578, -100.13404199551239, -74.3105958838058, -118.29615618809123, -96.1404126926611, -97.41462255459611, -51.102185778796304, -77.51620215800381, -72.60868536851788, -103.28731285968253, -107.25492556021366, -73.62657426728724, -54.01751321968455, -87.62739386858254, -137.8159265630576, -66.58868259374817, -83.63410803451515, -115.98616065544799, -84.46235736341376, -85.38634223058195, -75.75396117330557, -86.56622942916368, -102.48004599252235, -123.44988678496024, -103.58823642378391, -108.47120828090617, -93.97287897470675, -76.47801409630137, -106.54087160441952, -67.51447592692469, -114.32790118569358, -36.21112305387044, -135.03736930438484, -69.5192798468215, -136.17432781400657, -81.96025418972576, -95.51697631059727, -104.2671381435081, -88.177670512876, -77.11120857708465, -117.62358534300915, -123.4427551178902, -84.40803040980572, -65.25235723066335, -147.68040039525238, -62.034019277089634, -67.23809426509713, -45.12950378436007, -70.40690199551254, -77.57964349851142, -110.74110510435204, -79.60827874796466, -72.89820393878648, -162.4296485458639, -112.73052675596612, -76.05845792285764, -89.2386001062855, -96.1282834960949, -95.74942911494264, -42.07455240981769, -83.05746817359298, -81.74299803038328, -61.908261853171595, -81.22036619578395, -89.59651189803074, -119.36672538450577, -86.20832031276676, -57.721206185811035, -101.90783595940712, -129.72618339179286, -114.79460126985397, -85.1975027594322, -77.14591461642615, -84.21663765744188, -79.42882047529795, -102.73956894917349, -62.85803358350866, -86.7839228885291, -87.5579836151129, -88.92585410302824, -91.40766920656381, -85.94619881070862, -135.34691559701906, -78.74309897701573, -87.0444960284661, -79.7432568249107, -169.72717989212964, -77.64606518903057, -92.94282319519743, -91.09701925101454, -114.99463138982308, -131.27455052397784, -133.45181109592747, -113.97527914819794, -118.78151690770926, -95.34646398345316, -193.6588052995859, -113.9173297954999, -109.05760528009627, -86.70713261231754, -56.8421485707166, -92.96346193318982, -90.05782311433893, -183.11301041770668, -140.86918989870117, -80.63692478702089, -78.921902481282, -69.0523800973146, -62.30778160427559, -81.36754453261317, -70.41961195398551, -74.97720860599664, -84.80976748422306, -101.20426284799419, -101.05643285746811, -80.57805464008537, -88.86267198860857, -65.00348458291961, -81.67399709651559, -65.10843947136622, -69.17308954923169, -88.71695114910139, -73.63926843523393, -112.25154153675847, -49.61966949255476, -139.02893217443926, -47.51294456721018, -117.60630811518432, -87.78919798582332, -63.16863847298964, -53.8652768841987, -110.14433268429887, -66.68178146608031, -161.96969721507423, -121.24425378246092, -140.2898602553752, -92.05304391888339, -92.70616856846183, -75.70922781466959, -66.06135617642258, -78.4389062204647, -94.71039889135122, -81.76119600666736, -109.93697033529718, -64.1042364146514, -125.87229116603626, -103.4043177488131, -60.66111278711898, -70.23648874374412, -98.68316871673602, -58.451022051932114, -75.88055050484932, -91.37622263347761, -68.59621249842287, -98.64453751682578, -67.9134320712815, -82.61442076740528, -76.69234793109338, -77.95560933700744, -51.940492392587466, -117.48887215379553, -69.58012915499526, -97.17915774099828, -39.54880660317314, -49.14499472311626, -97.51221066278232, -100.03038337442528, -102.2892050108484, -149.49565054954894, -135.18490915590644, -161.73101704902766, -42.7821301737056, -124.6482748173636, -103.27139971208956, -74.63434616595673, -72.2074967360028, -61.62839072432567, -73.33264218880736, -72.58298362045494, -125.89342746037266, -33.77769284394237, -71.06462052615306, -144.67529240299254, -130.35453832321892, -83.88683831581614, -104.82807515381735, -88.79944000518017, -65.31278184612928, -113.91078570951066, -75.67831271693267, -103.45280407748058, -113.10093071388135, -100.62939155760131, -97.09748171991632, -116.80772273330514, -56.963322011635206, -122.81070161797007, -70.75248891736653, -113.34190341758365, -103.59325210713926, -85.29258218792853, -75.58021407871713, -128.81507854563287, -71.34135267635676, -72.63474140477717, -92.1021808677743, -84.00627367495538, -71.0073651936542, -98.11333193153214, -84.68642684691187, -82.9353032803756, -102.62108401432292, -149.0438085578911, -115.97309984346482, -119.3964931588738, -60.11035163567654, -90.30134633400792, -78.63506422707425, -67.72184835950084, -89.06289771575194, -107.82127751198122, -74.2034755051941, -130.1466571789395, -77.5320133231804, -83.33588009727428, -48.24499985994592, -102.21348928666228, -62.707234575190476, -101.16231081177271, -77.1209727272036, -98.43051298684978, -95.97002666219612, -66.50554209633229, -125.3802120325127, -68.20797692081311, -76.81834963220942, -80.81079479458668, -92.21752958203136, -98.70473474323246, -136.83965527016287, -100.53103232095206, -142.56763042993828, -96.85350790169701, -93.70659193458023, -86.33587578539023, -86.55558466715111, -101.8103292655437, -93.82913334430164, -93.4898779938909, -93.71511743368104, -89.35511373819438, -100.50024214625248, -141.42749580211972, -157.5685371474971, -93.5162531597734, -79.95221945357224, -69.35347347133697, -120.69938945793298, -165.85657017826466, -103.02084669016291, -112.76205502734882, -74.97141129235138, -95.6290842529195, -60.907935958861565, -108.18284329503001, -57.17302457732136, -103.90206494756092, -121.99144205536392, -42.2413317020569, -93.61906780555766, -88.26381992797073, -57.265550264619286, -79.04006962456533, -106.08756976645614, -71.09638344598348, -113.02692546005531, -77.49588093573497, -91.3687410133058, -75.28138601121735, -87.91468203529887, -122.40401606178304, -87.58789666641549, -83.16606624507851, -69.35307650437201, -85.64989197203268, -70.56761815735284, -77.90325431118856, -76.82515603424542, -93.77641295320551, -125.56139985076759, -71.72747448655394, -98.8433640553206, -51.844117766817064, -75.51769431162423, -67.30223874778318, -94.36093088768712, -80.79294744913473, -80.51225281436895, -63.02231860906229, -64.84737392924957, -50.42290276729421, -137.47426481129682, -119.79372230603455, -69.9013851173589, -68.13662252814873, -79.73093752102515, -79.63135226128787, -60.92746065378604, -54.10724219705017]
        rewards = [375.27308161433723, 306.6693514768444, 265.7909342957096, 406.9267386314135, 445.15937870916173, 264.4403842329684, 342.3587424664886, 435.89241956302607, 347.35848419491765, 244.46087810861178, 331.3276657469517, 357.81187821343156, 279.12264420301307, 451.565187505841, 295.06709214176567, 393.2690964809012, 343.2193060858441, 316.7733814337985, 518.2478957394738, 172.84154989256265, 350.75628338798765, 259.56155497345844, 359.5923629826576, 378.7371846131052, 373.5831046352233, 340.31865378066055, 276.5067489849742, 331.91580362947457, 355.096471226904, 326.4924420565719, 330.9514243729766, 12.288409768362595, 401.1349206465958, 305.90708835222335, 239.17936801823674, 360.62553420631457, 349.0766037362859, 243.26587990347153, 427.98854232402925, 378.1094116151636, 427.13393602127144, 274.7902258979717, 311.7738381590822, 361.2546984907905, 315.6615592855496, 72.81068996469256, 353.32479052080157, 298.2523997935985, 295.5828621690839, 358.6581022278024, 282.29699770863215, 25.466388572921872, 344.50934613998743, 226.5676405811269, 392.0333382182382, 360.4299489603144, 312.9928830786123, 262.0974470653715, 391.240275041017, 266.82459473427195, 306.1858244026191, 204.74024401330536, 309.55495377863286, 71.01637285316346, 309.3746064656757, 108.65848165360492, 317.48468299274487, 5.7714446298428665, 384.11535250290444, 379.51406766963555, 437.3100179206932, 284.03004568343295, 281.98928595101313, 391.15581189655677, 420.92370370686734, 322.6294356337405, 225.38752331556492, 307.50376149106125, 359.0758644795897, 362.5174898680845, 275.9914146772138, 119.62263602570411, 449.8905011271924, 334.3997872853582, 323.53024831552807, 263.9618319801421, 326.8062693520552, 293.2411244913157, 387.66393907406575, 329.15249450286507, 261.7044712926904, 155.21562513825174, 283.7220095224476, 22.48818847799126, 352.92198036615207, 373.5881837787034, 447.7512000552738, 405.0477539919583, 184.71797781471508, 274.1126723220823, 431.6190408292215, 337.88683477638637, 251.06212444604802, 297.5069745287276, 368.4223332653746, 290.48194912519205, 362.58078738716534, 330.1097902097998, 409.958847607256, 222.87087767177513, 348.16822918910873, 243.0057553708568, 333.66161018685, 334.64199474750245, 278.18443745420996, 239.89918003190917, 130.9062900325808, 385.0119824479958, 378.03462097653255, 217.76042457033776, 322.9783819715733, 246.08008000355372, 335.4247587446904, 29.3901887335672, 285.6690941607602, 435.07989760145466, 309.54192647688274, 378.72066555802735, 175.92139097148933, 324.0449468296102, 361.85428068549885, 369.6274181364839, 325.8435884677693, 233.7604382441952, 422.35952631117067, 289.12046601569386, 274.84409196096976, 333.7270584176056, 421.35512249586384, 251.3525244639948, 399.4331889063716, 393.4786906304653, 334.5023741063088, 277.5224150925807, 266.32633475784365, 291.1968411185376, 367.68550563260845, 255.8026230086959, 300.4090704924382, 353.0557131327594, 1.0, 386.5187190185479, 334.50247020459074, 364.6420111105384, 379.2491216390703, 170.85094378091523, 247.26807724508046, 305.8781761939971, 350.0515024457021, 387.53812419605833, 327.00562308131197, 95.81818151483405, 1.0, 343.7707639034369, 236.30742310675265, 368.04527848746045, 335.4786059485572, 253.4909982739125, 281.6901186489263, 294.89891650500954, 50.68314015127841, 305.59815215199353, 322.1600850041839, 388.7574416478832, 342.70910127983836, 453.4338297392576, 275.5351762584992, 355.0220801734772, 362.16402296922075, 389.4121472038448, 336.82828027973454, 245.7847582426923, 376.28914359796784, 250.2195522643338, 336.1314477304087, 327.52262532649127, 272.21367375703534, 333.60647052651376, 454.20636652108385, 393.37248229382806, 343.93085889727, 408.7020596360608, 385.82385305859646, 373.2927894593857, 225.1238707906216, 334.39898603820393, 252.0761569746549, 373.47500675398896, 349.1527787869627, 355.6044281135263, 298.57873059717963, 250.96383309640822, 355.7825259540824, 387.84275065758914, 425.2308616361415, 345.0194971110516, 401.4526965220242, 234.63298540622952, 330.3505317927834, 223.5195810611866, 326.8191439660641, 337.1632510612973, 433.7900448536649, 430.9752914346106, 364.36902075175914, 295.3184010054338, 268.5607283965047, 290.28127678193175, 300.75617596075017, 360.45679609149846, 370.9850767062933, 327.8473673391734, 40.83897306366042, 390.9453791361109, 224.39342913778898, 341.43406904926354, 330.007527703392, 311.6722749780312, 350.8808519006953, 384.2142131156221, 323.49667811517503, 385.36921757456776, 376.2110975874355, 322.9613766045891, 409.4759498675893, 306.56965737147186, 297.0649817847392, 417.1146599545066, 335.2878560702709, 435.2937804210295, 417.8820933006103, 272.85204727123465, 426.55947074374126, 364.17319198771895, 447.5150485240558, 341.19467608010723, 352.9689039445965, 297.84548687155564, 401.1409543224677, 319.3612735273004, 76.03593200376716, 163.15212054415176, 309.63151250784244, 482.7332783877072, 393.97503491078953, 239.99361426482335, 466.1057354967911, 339.8524184622977, 404.02287065009523, 328.6431839116743, 257.16832778013304, 358.3498969326969, 292.1592649137029, 209.61999938995382, 289.9222599350402, 518.1464009454918, 349.4006463655247, 464.8680621165787, 377.60750315624284, 530.3314668215085, 316.07555442738044, 333.0013013885923, 396.0218213061862, 307.6906427657366, 406.0415178739132, 338.20547368880995, 219.83137495242295, 346.9396470234665, 301.26910359071974, 321.4782633091767, 341.19670808499853, 295.63172786436223, 279.8763385124377, 261.6667336834442, 283.4884939551102, 249.86082076222885, 397.148349836285, 402.93116044006405, 373.5698291359302, 234.54107020978162, 438.4432122429994, 330.2008967362168, 413.7343451362555, 338.65555255772847, 314.6556564036973, 267.58653006009445, 347.79747415723887, 318.9258943033613, 394.26864598980717, 512.9057373390176, 401.02847844589496, 443.0743201002913, 405.3228929824662, 339.69188481967444, 290.6781831184105, 333.5871185686828, 289.9624860083048, 283.977976638682, 324.8520213672347, 264.8684055260628, 300.9843668574193, 243.15857703101318, 382.8546878239169, 385.4124471410163, 325.2761333253795, 243.82134091468217, 285.58402619896674, 185.39122204248622, 305.73371764931096, 313.62341357842746, 103.08490419172765, 250.79493171753487, 265.9356575664064, 250.71410339300903, 319.09187872354215, 354.5521275787281, 196.79341547861333, 281.7688949784115, 96.07258860226625, 330.78666888916314, 335.5703684828708, 374.7635651730643, 417.24296409594297, 445.5407725280404, 300.8266920324055, 344.63528329876965, 409.8324536363658, 324.9302826265945, 387.46008920603, 266.98644132678453, 272.5836877607585, 139.92303213550824, 205.3562777030137, 354.0279359394625, 450.4044135126953, 293.09228260187894, 307.7583596399441, 364.3407927096813, 412.58542951958583, 419.67824024789763, 391.47036505517553, 410.93858652564944, 291.7290656323451, 239.2781604975113, 196.82277400835028, 21.92259795999528, 298.1904046855861, 450.74022320064006, 446.48210929856975, 393.22496777059746, 342.57525130677857, 179.2313582513549, 91.06498064117443, 108.51808068577884, 335.92012550176923, 305.6793223755311, 410.4809485215636, 313.0393356063885, 193.29185974257916, 310.48683372867686, 271.2090371959697, 1.0, 201.53866925089116, 338.8715629646618, 443.39698232824395, 303.68118818261087, 327.9868181575954, 285.3221614483409, 321.3468172009932, 260.4295784651794, 285.8764032914362, 396.1428618315126, 323.26368425087253, 392.09559936328685, 332.36819104500927, 503.87766769852954, 276.8419287086599, 401.1119523745461, 258.85597315926077, 427.93298737769936, 316.88332134938037, 293.95242750132616, 315.18504309287266, 280.14865098348344, 286.5489819424155, 348.2050427091857, 327.6294053961051, 363.1715367131811, 347.1502224748783, 296.622274280588, 356.54933574833615, 389.17704806135373, 372.68687117916494, 311.90274982545185, 264.0698239458393, 283.9343666491167, 354.8242411761794, 396.8708120966536, 364.2269677296878, 421.427528266782, 194.00195334394823, 374.6686017319253, 373.51971851814056, 383.9095523869522, 340.61765188423215, 375.96834405050987, 168.7803159366685, 326.58978437934417, 411.2798471286922, 363.69258485070446, 74.83303965190157, 381.6837924647665, 353.74057232936775, 390.82525889536134, 307.3071819134001, 396.7806288925244, 344.9328596186143, 345.82214172626135, 319.5579493101565, 214.07741894404472, 262.94937199266667, 328.43963383222854, 257.2124854804513, 329.55123265069096, 167.42607240154848, 209.83657468632677, 313.6500648365562, 333.4682433168566, 404.20308705023126, 324.5755250201353, 375.6628959108059, 389.56910169607926, 328.10248796880796, 214.6940244210992, 315.2727441445422, 384.1852080971996, 211.51800192260015, 239.5911509602833, 286.5737520208431, 388.97365364657156, 257.91679141073917, 354.82734634252574, 453.4738235965699, 410.02857098528284, 349.4536427851299, 409.5428914945022, 33.10611420402449, 245.03875687051817, 344.69433638499413, 22.904780704108994, 385.3089968517651, 331.5891840457215, 343.2321488760567, 347.4759902540761, 341.2498428117553, 242.8607168543639, 388.17750332298783, 233.12922338323568, 389.31256578716136, 174.6931428802303, 169.00431275207382, 358.9930905964472, 361.80685773004075, 461.7970019111175, 435.7016009112356, 413.8230683898929, 461.65949221889457, 454.3624518017595, 320.5740971402174, 201.3293037910862, 307.3965820163804, 337.1181629413488, 411.6867167062517, 380.4728184602212, 380.67152042582666, 399.06404712752095, 306.93720242048676, 361.29289244611095, 359.70108029863883, 353.63059229032433, 378.9531353164383, 238.12225286249202, 246.6278170638373, 380.82236927922764, 292.55233012913595, 387.27529524182467, 98.44082537187691, 344.4339091061415, 258.6164376023079, 371.5389166038458]


        # Get a fitted curve for total_rewards progression
        def fit_curve(x, a, b, c):
            return a * x + b * x ** 2 + c


        popt, _ = sp.optimize.curve_fit(fit_curve, CL_CD_list, total_rewards)
        a, b, c = popt
        x_line = np.arange(min(CL_CD_list), max(CL_CD_list), 0.1)
        y_line = fit_curve(x_line, a, b, c)
        y_line = fit_curve(x_line, a, b, c)

        # Plotting of the rewards progression vs CL_CD
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.scatter(CL_CD_list, total_rewards, s=3, color='#4292c6', label='Total rewards')
        plt.scatter(CL_CD_list, penalties, s=3, color='#ef6548', label='Penalties')
        plt.scatter(CL_CD_list, rewards, s=3, color="#78c679", label='Gains')
        plt.plot(x_line, y_line, color="#045a8d", label='Fitted total rewards')
        plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color="#737373", linestyles='dashed')
        plt.xlabel('Objective value')
        plt.ylabel('Reward value')
        fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs Objective reached for {aircraft_config}, 500 000 training steps')
        plt.show()
    if ANALYSIS[6] == 3:
        aircraft_config = AC.aircraft_config
        # Res = Results('SR22_1_272_000')
        CL_CD_list = [30.065068220423818, 30.562174190409454, 28.889417491849763, 28.864424933808888, 28.291758401049307, 28.171016293971697, 28.08, 28.304969598400444, 28.08, 28.08, 28.810752777420543, 29.139705420283356, 28.1260635808481, 28.611271371628703, 28.575039389514178, 29.112381260925364, 29.027380323321516, 29.67353793563832, 28.463594107898953, 28.08, 28.08, 28.08, 28.746402292306804, 28.228069477418614, 29.281071574081107, 28.365571626963693, 28.34813432507483, 28.934320654857125, 28.89390573873199, 28.908603857407115, 28.92727354130781, 28.08, 30.555608035896554, 28.214848117503028, 28.31004701266069, 28.86460215566915, 28.08, 28.08, 28.140811457423215, 28.17319906059334, 29.47345338292157, 28.577857764242513, 28.299203894382014, 29.022791313340367, 28.08, 28.08, 29.398859806627488, 28.08, 28.73792808003002, 28.12860523546228, 29.826925190166037, 28.912459640738163, 28.151085827772675, 28.752645921181795, 28.90528928559309, 29.561104768283766, 28.142686772653317, 29.102115362910872, 28.39760177418334, 28.160689668843677, 29.3268490756333, 28.411954662736633, 28.17307422203399, 29.29652427321698, 28.295456103385924, 28.43835354935911, 28.08, 28.94902751756511, 29.512908888289815, 28.379512571145508, 28.08, 29.139491788281216, 28.701890787466503, 30.481741502068154, 30.056052599887742, 28.31699315124107, 29.506978733706024, 28.436143531202394, 28.79022810901229, 28.99754359772371, 29.043998065931778, 30.09357402279262, 28.21076511867283, 28.182431306319234, 28.08, 30.393770821695963, 29.823069026500555, 29.46402070466506, 29.13502503494832, 28.80005352208926, 28.63072660674481, 29.65457648029553, 28.08, 28.523990234120962, 28.23591315638483, 29.083879958958814, 28.230770672733787, 28.770277827576326, 28.182261179753258, 29.364203614158132, 28.154011053354765, 28.175661213085075, 29.051053981884223, 28.187612725682325, 29.422937511905957, 28.108505043001763, 28.08, 28.51587080091352, 28.33927716701043, 29.709084662676187, 29.905763767197417, 28.08, 28.410552885084183, 29.759663628184555, 28.4182717174819, 28.08, 29.26393825916392, 28.75719227727147, 28.289362781296052, 28.124882137477663, 28.343462456908245, 29.077178996249213, 28.432394868775358, 28.130454093682836, 29.32575155025916, 29.35524594581219, 28.08, 28.66197720244185, 28.7008112275602, 28.18461822167999, 30.33686746263084, 28.1220816579973, 29.323927140914616, 28.57843762239944, 28.413389133901067, 29.33715051300642, 28.803470795550034, 28.08, 28.91176131410443, 28.628180864218855, 29.071137956260273, 29.506697371853342, 28.967035723924443, 28.08, 28.08, 28.61298858567508, 28.905397677076653, 28.213542716993494, 29.325394858523268, 29.17144966100276, 28.833871053453585, 28.585850738344746, 28.47349463244012, 28.20326676736857, 28.332984288005576, 29.694163495328844, 28.40549605515967, 28.270445551403444, 30.365976632385262, 29.247963998658193, 28.553992575827746, 29.924589613941137, 28.186439060001803, 28.08, 28.712473268922963, 30.79422379439861, 28.08, 29.188811239859568, 28.08, 28.43684292811472, 29.287688223103203, 28.685959972218473, 28.483115836290278, 28.08, 28.08, 28.725572501388882, 28.22498346022288, 29.046992262345206, 28.276943433901565, 28.820960434510212, 28.153941260633463, 28.25334003585269, 28.699847142920422, 28.17947997785284, 30.569517098744655, 28.636108131209006, 28.08, 28.355792339305175, 28.968322063913448, 28.08451943084136, 28.08, 28.20737269310555, 28.08, 28.77473427120445, 29.6097533346697, 30.21519811959238, 29.720081549870667, 28.08, 29.325217531773564, 30.29223690148332, 28.08, 28.08, 28.08, 28.36461115805341, 28.388478536842165, 30.49981623094233, 28.11243173598606, 29.70661731435142, 28.08, 28.35794134281972, 28.792174839708604, 30.089539750352195, 30.151129596284985, 28.23417748140621, 28.236771366205275, 29.367004133386324, 29.139434532855518, 28.38043203533674, 28.08, 28.08, 28.08, 28.08, 28.135396009730385, 28.961277318549104, 28.08, 28.08, 28.275186334827847, 28.921159116465926, 28.956358563805654, 29.180153499075725, 28.113686424803515, 28.886899934113377, 28.262361769453882, 28.89171696295648, 28.214682328533303, 30.226884500459708, 30.077438282616956, 29.166058451835656, 29.573299304812686, 28.959740369806347, 28.08, 28.466041548015905, 28.60236586317169, 28.64539845856874, 30.303357665522775, 29.059145050918396, 28.13098287604206, 28.08, 29.969124563992885, 28.08, 28.18570008074095, 29.66693286436717, 28.327892889453945, 28.699849513977114, 28.71832348658316, 28.122253662791618, 30.529208224056113, 29.792639546585033, 28.544481032100354, 29.17728221454263, 28.08, 28.097041410857226, 28.08, 29.621986725742886, 30.288216990937492, 28.08, 28.572943449207195, 28.386734701272744, 28.09601702862297, 29.91805229402813, 29.06318953158025, 28.806519698882965, 28.08, 30.120203533661623, 28.64806373274175, 29.274496466034652, 29.643592823833927, 29.238021308579018, 28.687566898014076, 28.456763573800043, 29.27713167281294, 28.08, 29.407260412495347, 30.061184868025926, 28.628904897377446, 28.750742690278788, 29.84230576045013, 29.74398599835497, 29.10033166106637, 30.75836929690841, 29.186648577364824, 28.08, 28.74415133148236, 28.3780509689432, 28.98362464165963, 28.160993708494217, 28.08, 30.165919510123654, 28.14105442132614, 28.08, 28.644097893507045, 28.40692121425773, 30.032234073871837, 28.187662230207625, 28.44423163119755, 28.386212572126226, 28.08, 29.23417337541393, 28.08, 28.797634523950215, 30.11458738077144, 28.202040201344523, 28.90353576621007, 28.63156756008603, 29.94474176466695, 29.89406066321093, 28.98287799772743, 28.330284970092727, 28.77426470247023, 28.08, 28.08, 28.331508135362743, 28.175871114599758, 29.5346407433062, 28.87404328318819, 28.213330095093728, 29.664958460155677, 29.577713483609507, 28.08, 28.10396473632702, 28.940779936863215, 29.452637029025624, 29.69580881231414, 28.08, 29.679055041261044, 29.896438363875305, 28.296104779623843, 29.396758870286014, 28.88760930103675, 29.19886562817103, 29.464566440739702, 28.085334936683644, 28.08, 28.319615095098552, 29.235136625540328, 29.19276870654174, 29.07696809907827, 28.203451444306676, 28.098955534920773, 29.788087379682192, 28.08, 28.08, 28.2813998326375, 28.98527738794576, 29.428194621454345, 28.08, 28.649628073854043, 29.045215762724375, 29.278349745850853, 29.0846976871263, 28.08, 29.486018742195306, 28.800201499218513, 28.08, 28.58981576746028, 28.91070933892703, 29.87870007463071, 28.13688686381835, 28.17178999689584, 28.08, 28.3225623738593, 29.046428541666454, 28.11059778300609, 28.08, 28.08, 30.03918782355993, 28.08, 28.08, 28.867697425857525, 29.652382200871664, 30.1570878251254, 28.08, 29.69787852216682, 29.7148601511106, 28.08, 28.08, 29.103573759330715, 28.08, 29.85677498046664, 28.788510891873266, 29.296936811341848, 28.08, 28.08, 29.76107112322786, 29.797277353988484, 28.915981114594576, 28.08, 28.101826213902203, 28.17322637747642, 30.005922116182195, 28.396061923349393, 28.09614370933293, 28.77098977675643, 28.12828263224249, 29.05461620318988, 28.702387090784885, 30.397965551881107, 28.372931931982897, 28.28819345290376, 28.08, 28.809515531780594, 29.138842649557272, 28.130468887701294, 28.083957974412357, 29.31389293163465, 28.11821437164118, 28.08, 29.20242921762807, 30.277010514097494, 28.10717766018422, 29.6685759560119, 28.08, 28.72544545208972, 28.50163093577619, 29.39455817166369, 28.211143633486273, 28.160790369995503, 28.156804311155415, 28.1635633222583, 28.850864625747214, 28.476200892618323, 28.139895989998656, 29.357802480299966, 28.4178720016058, 28.630367876145105, 28.08, 28.08, 29.840732490886616, 29.62062787195748, 28.08, 28.14343920203837, 28.08, 28.435021568716255, 28.522063147589428, 28.08, 28.185590123429474, 30.270318272337352, 28.40187264784219, 28.08, 28.08, 28.096102675310533, 28.08, 29.68329995078616, 28.37199790253803, 28.08, 29.96855722363392, 29.935977456852726, 30.22299971101047, 28.496367152883966, 28.630499076419827, 29.502771838281404, 28.18214907773829, 30.090847584078105, 28.38020822161493, 28.849496781509384, 28.68870214204495, 29.470678415363167, 28.951177718186216, 28.08, 28.672587281629895, 28.08, 30.071871804646015, 29.691577507713497, 28.54433252715667, 29.26316037960872, 28.08, 28.260652765700065, 28.828464082609646, 28.47708017031476, 29.082656555624677, 30.478277959885162, 28.165129383353, 29.12539881312127, 28.502035661913485, 28.560769832361853, 28.446106052528105, 28.08, 30.085473891179753, 28.54885263180774, 28.14971058596191, 28.48736664093752, 30.597235446637765, 29.71927807175763, 28.138569818030664, 28.86433002053218, 28.661507363083395, 28.719881111006544, 28.08, 28.08, 28.452117761730587]
        total_rewards = [38.303319484627124, 277.5800966395075, -65.48041902867672, -177.52439665205108, -49.550934800879936, -56.822682142480666, -65.0385880896223, -69.15778255512413, -55.35004745515313, -46.36220239750543, -44.12128390281192, -30.226270587933236, -42.79081694318566, -34.201231225725046, -59.694036002988575, -73.69004081123963, -14.096006367686467, -11.185525100332555, -60.649460659832776, -42.239282097595805, -63.7199931884232, -38.83591849472452, -44.38058660798978, -58.636333981254054, 7.6528810290096025, -41.78326822818357, -57.02744525726709, -23.826714037435135, -29.5929395751796, -57.81499487950616, -49.687436274118475, -47.93886556487673, 174.29537328077078, -43.470787426238616, -47.843401149696504, -51.7880854603782, -44.47904922589572, -31.50378890198889, -35.05268398446339, -44.48405744849075, -43.6302508378307, -69.00719331207019, -49.689317971278, -61.68993199242121, -57.24660949310622, -55.70709109199103, 7.632371847519796, -38.26597703430375, -65.71741805772437, -45.889268268386886, 44.088662239100444, -84.95022184431787, -70.66220018579223, -73.86546754625067, -10.659007577775958, 2.6974814440457138, -51.795025717545215, -35.350369122128015, -61.522349397209716, -95.87705259592339, -4.699712462670078, -73.64962958318618, -49.50248194804644, -23.42661208296804, -38.90661245097421, -80.09574129958197, -58.552648430604926, -52.054292448340014, 8.622123145658197, -70.45206793601939, -49.406835324519214, -11.046279100259088, -63.408950608196335, 213.73430802251562, 60.85326080340207, -61.5217523856133, -49.08094076918673, -37.80685047429242, -35.28184117675481, -34.715264746298445, -59.85954551042127, 123.49683193876632, -38.08840621149518, -53.043716314576585, -50.04957731496775, 131.08035816275964, -49.96400687991205, 11.636620520466437, -33.45894388817797, -52.57086074808281, -30.414504778181744, 27.42970955747478, -41.902900641649374, -43.92401382519141, -86.17774677651573, -23.570041879019826, -59.00107425253914, -22.412136526098983, -39.41709024642866, -9.636853252401231, -37.268184589170545, -39.20870815660238, -18.237458443264956, -36.68797737238646, -10.355898442217683, -34.76570050221518, -63.75516369643127, -67.04209970633674, -49.79159749121363, -47.55455233915038, 55.78362410149389, -38.186556550563566, -60.45258706525333, 34.50966666604723, -38.4689507224009, -70.36327024258159, -35.22474518698384, -38.63212907205378, -47.80838393928912, -45.834309787205235, -60.05086065718628, -46.32333394064665, -82.56016138932533, -45.56005484347494, 14.861855379667517, -107.36627761094998, -52.66032518933083, -71.79267359939928, -79.70308697971132, -40.59488275568782, 53.4856144392718, -48.73692482022565, -42.11152311622406, -62.51080887632453, -28.120463013745546, -18.703675208043183, -23.918740799350633, -43.0747592077602, -70.73967993344007, -50.89190678775279, -21.534762652583577, -41.24292455238284, -124.81130627620233, -52.08180142488994, -49.045135338239206, -27.52378724395954, -63.6464952023196, -59.79353218156281, -31.671026643269762, -62.81408909152442, -12.130996785189355, -82.92841460046866, -48.87399758091015, -32.67341840127761, -44.91350065970532, 1.4530787428476728, -41.5061498953685, -42.521708141350054, 224.6433098686149, -56.41085448609911, -42.83293224916832, 59.389188033701544, -48.290568590278156, -36.70170333613745, -56.255579077171646, 279.44034112226683, -70.82800786468465, -74.89073872019512, -41.281467534961834, -49.5311972339084, -70.18719706837311, -72.76715185232142, -34.36590219241651, -32.91605640054588, -37.91070123958084, -63.680929550656145, -37.540511268701124, -76.32694195770127, -51.66858988868978, -81.67831975028054, -151.07652189204362, -48.74285727269222, -33.22604867221324, -90.06347947817684, 227.77736387561168, -79.79267735458916, -40.51794799211572, -71.88515500620166, -58.70716650944601, -85.77540383386152, -53.1767232682009, -50.558182372017214, -63.95766544473698, -55.75472822342283, 7.3065391215992115, 91.69909659778466, 33.2163819185166, -44.82451813755188, -13.25634100841489, 141.25037705038207, -67.66775818528197, -44.053493254785394, -93.64351299987752, -105.7380077524372, -38.32511740065905, 79.65395825249941, -40.600785592631325, -20.52736794284624, -46.410500489211756, -33.212901769212266, -43.2063977733152, 64.02211162023588, 107.48555082055628, -97.31310373086136, -34.64869457034986, -12.907650817738155, -54.8947051976759, -44.03023086743967, -36.54617304003958, -55.412965001424084, -30.169353812596572, -43.08450300793773, -42.77143479514999, -24.868002533281775, -45.283561147182645, -69.0046510716963, -56.6381644725833, -25.57888873096514, -63.52287479869608, -59.910815664757976, -44.60585710927255, -83.75713582524472, -80.02936771164848, -48.6359131175101, -51.63739910981232, 142.446992011207, 82.08472076910436, -44.745091494386244, -81.70966913318064, -27.613671493266523, -41.04519362764758, -64.00755872646752, -48.1849829725038, -41.6904431104588, 86.06614579339058, -47.432728834938125, -45.91636145405977, -31.589671898287715, 62.464786876813804, -45.28338106461178, -37.17244817677944, 17.068301224318247, -50.985510949139154, -119.69823014541393, -65.11216110523922, -52.9171321084749, 236.89619787531223, 31.236546913890486, -51.44716826299279, -33.72053379466739, -47.206477700174304, -42.16412051757852, -41.37894025412056, -50.80760117000503, 173.6100510784148, -38.23248738903292, -45.37408059089452, -80.62042122671103, -72.02651095013425, 50.971904397236344, -85.57512137837233, -32.62139808093879, -43.69685572756444, 83.68588958438986, -94.53236700752096, -9.634169634188272, 69.87627914234204, -58.41331955876234, -98.7113938069195, -89.11983484618084, -94.75971935499551, -58.706242523569756, -8.087056530135204, 125.38380774803854, -32.65136067160374, -28.9996495904841, 10.196557520780658, 6.85331225417189, -62.542156926981946, 285.7689692901822, -55.96450248250056, -51.03031574898999, -49.282178876696484, -29.062212295872026, -70.88361069295098, -43.829316622781825, -41.84836950148565, 112.40999407135413, -33.29573756012573, -39.758765803453954, -57.70959609418105, -41.55425787070687, 88.83277603799752, -48.365282982214225, -45.994075138110595, -64.9732448765231, -48.313757893151745, -19.11031688577045, -45.01481552400315, -90.10806421712962, 83.06127292761616, -40.57185648607365, -62.55718525164813, -27.341717892406287, 17.221553468486064, 3.962176009020169, -1.9721724811458987, -52.94821874154993, -45.1612431605308, -36.209534721472885, -35.276783204827176, -47.619264923336914, -45.083001302721236, -1.1165059786378748, -46.20365838019234, -31.064937013233592, 19.489936451303677, -13.179938830917031, -38.569023113517, -47.46881447642484, -147.48277356293983, 14.193118518431454, -26.651229692490304, -49.5553962635706, -12.623540023522345, 50.55929436491482, -43.85649821390046, -41.771837164751034, -29.640661431475824, -43.91717855938346, -23.749806596942676, -43.25699828376438, -50.99087542344608, -80.54106717444805, -30.05548191840601, -20.180909558089052, -57.28113140201906, -44.403374174089336, -51.82858457988068, 33.67783942947601, -61.47262627959323, -48.610048477209745, -69.79944366993543, -50.58660938210163, 0.09712089976419591, -46.111003398029496, -80.00753254807738, -62.812863359033365, -22.15720804716248, -15.988258911122106, -52.23803725575029, 2.42609434442706, -52.89538787263786, -44.70525879617925, -41.27976135432126, -27.404597125032144, -55.433727610613445, -66.81379518725585, -48.74991030639207, -78.92758723681668, -31.819169286233155, -26.905293852499007, -46.09755473192925, -43.97958283513867, -39.62192912417968, 74.48966752337081, -53.673150049490204, -45.941980908744604, -34.46863514491495, 21.033535931119154, 70.59958237241479, -36.15450600415751, -12.661294980284211, 40.57193340131708, -60.21496198066396, -37.24836733413158, -26.20230891971947, -62.329296924462085, -9.217973259568353, -19.854313092965533, -65.04018652898988, -50.57367706012236, -41.16096352189363, -31.73490386020073, 52.449188748871116, -49.12975587457425, -43.11304844844035, -110.28484542962553, -48.03521235933221, 40.46449995740957, -82.06713911742733, -51.98793124067642, -54.22451354352838, -38.87878329630643, -64.30233179384973, -55.177327989319416, 126.3893605242194, -40.00515975258737, -42.41451246818875, -59.71631192276033, -39.51516488361536, -35.02198789602113, -95.19832811667598, -55.79868460669832, -40.53801822689334, -48.1372086955048, -59.95624354269167, -42.53481706287304, 185.32062700181874, -56.879804373542264, 56.130147853808324, -33.598240447486425, -84.23599064558752, -40.63160383201846, -62.92465485003795, -48.77082146435775, -71.01135442607666, -36.43629243520994, -53.5106121658197, -100.75274612520994, -40.19245702108304, -50.34278766321203, 10.346579294509853, -53.84360112143165, -47.57678469907834, -44.184343414637574, -35.74549588023434, 19.84530456691415, 27.657543501988293, -33.05031584129924, -36.12156857138611, -47.436195349729154, -46.682821132943836, -55.9167765314735, -85.2887261553528, -67.60056323668192, 170.9215973888048, -36.15649262685584, -36.38257049520614, -44.37949974401272, -63.405572454885814, -40.364553451407346, 25.47443257938255, -44.73723456567666, -54.28042283723655, 117.52871421520487, 71.90756661418075, 62.81126336253911, -68.38933841106473, -62.6999974678012, -19.79949486621834, -48.09380423225112, -12.335274884858705, -86.29784503707069, -58.03998866555109, -44.7428691942148, -17.03169142170513, -28.035810602301886, -43.10940776099324, -60.86405219562242, -48.11428543682329, 55.086450377925345, -14.984505468553138, -67.88753947085344, -8.019698253227642, -57.93178288289112, -39.22502343824322, -55.599668880629466, -36.80054208684571, -14.368488401276881, 241.53021111117832, -67.20621168636993, -47.97580482273217, -28.183240812033084, -50.76010948619446, -56.53536419291163, -41.460877820446015, 51.434049792783426, -54.35000021767175, -60.71851891142212, -45.79816953736844, 279.8153062263037, 21.863557948764242, -52.68181460540592, -120.71300203602974, -35.579355290067596, -62.48434683437714, -49.50413994676441, -35.12673397039652, -34.435629620029935]
        penalties = [-139.71159356575737, -109.71199929771424, -96.40573127742216, -202.14577094111218, -59.838914496452816, -59.445548212080716, -66.0385880896223, -72.96656945028323, -56.35004745515313, -47.36220239750543, -71.41742367055313, -86.00086682862884, -45.08928811401614, -51.60653597201312, -71.03714132764287, -114.63997096815046, -64.1797773305647, -132.5302824564603, -70.6929171668322, -43.239282097595805, -64.7199931884232, -39.83591849472452, -69.38885974329473, -62.84119509451157, -53.562468139833456, -48.731850703012036, -63.95035834317269, -63.27690101658834, -55.283634306514685, -90.44924582642922, -77.02185861663311, -48.93886556487673, -151.743517133154, -46.44387541124158, -52.89786772496565, -90.59466664564931, -45.47904922589572, -32.50378890198889, -37.453852459268866, -49.26576331623623, -119.77322508820929, -76.50740444029437, -55.628648504911695, -91.40124118299416, -58.24660949310622, -56.70709109199103, -65.05787319920466, -39.26597703430375, -91.51537226115796, -48.205184661793176, -95.22327515319836, -125.4466681276418, -73.13704296018152, -100.11072039399792, -44.73987045068771, -104.49468696419214, -55.24856583386849, -77.39141848574896, -67.80993384015511, -98.42237230517777, -85.76647297263636, -84.22308492781791, -52.141022960681944, -99.19861624266422, -45.00213961570265, -86.69685290970455, -59.552648430604926, -87.42350237732134, -71.41193167091357, -82.18994694909684, -50.406835324519214, -63.360242101262145, -80.94890073055336, -157.92975931092056, -102.17752906398684, -69.53756594402749, -98.44749929662558, -45.5981760397492, -67.51866678860547, -73.46025592347027, -99.4619892892757, -102.38300548040233, -43.12296405749028, -57.96133650428148, -51.04957731496775, -118.53693087279457, -229.81854132072465, -65.79513163043399, -87.1270874831177, -83.3152561672743, -44.154738922221384, -87.75423778489768, -42.902900641649374, -56.8928082260904, -91.65717993407705, -91.4639148939526, -63.17419881308722, -45.45658845950658, -44.32715192072042, -86.62156320757758, -39.76432784271404, -43.02325319956405, -57.63013862393234, -40.44868183558506, -97.8177820035626, -36.946489008362285, -64.75516369643127, -81.35943351420791, -56.62227235555839, -161.67642686628758, -106.74177667811132, -39.186556550563566, -73.45290470955361, -117.22738983382972, -46.88808164887464, -71.36327024258159, -94.82216139203612, -72.06234260199274, -56.2925918797983, -48.12470750223113, -65.91406988268373, -85.79624809460516, -90.41164967811483, -48.96268360287263, -50.02949912846761, -203.33224935040707, -53.66032518933083, -86.03304296940547, -92.9757622320307, -43.32273329455687, -186.22779908780453, -51.00827535799573, -110.71149793966602, -83.42713942970171, -37.082631705025584, -73.86955196896075, -56.52209482867193, -44.0747592077602, -101.40968707575055, -69.04321259474948, -61.75674025822363, -135.5150399216363, -169.24720327072657, -53.08180142488994, -50.045135338239206, -40.25347118828057, -102.14556694959035, -62.75568307445831, -88.73529036952809, -110.66641204164348, -43.24106418560954, -98.94926183535327, -59.10377603600025, -36.67647498265949, -51.571295866312504, -95.42169715199118, -51.83171953886443, -49.3958523236364, -143.91318590316718, -127.94814961288866, -58.43939860310021, -134.69622635192516, -51.03272167337877, -37.70170333613745, -77.78266770248797, -159.46591197263572, -71.82800786468465, -130.17229816388513, -42.281467534961834, -63.44142561281189, -120.45711188377712, -88.01329089591471, -40.361482447667676, -33.91605640054588, -38.91070123958084, -79.29684113837712, -43.03196728648935, -117.47453295747484, -56.24282111679417, -110.78635404434579, -153.5721552517538, -53.205784062362014, -54.25623678831391, -93.97109241545165, -95.80105208119666, -100.92449202227253, -41.51794799211572, -83.1540641962143, -99.47759472928296, -87.80324029659856, -54.1767232682009, -54.513866655164996, -64.95766544473697, -83.42915208493076, -92.90428306300619, -94.3312711088238, -89.5318296223123, -45.82451813755188, -92.039194901535, -127.36831515790743, -68.66775818528197, -45.053493254785394, -94.64351299987752, -113.78246510049, -45.825032403418476, -213.13304607510847, -43.86794770750085, -125.57259713517918, -47.410500489211756, -42.72196310718598, -65.79873041578314, -87.83501638754947, -132.60403951375147, -100.4519026117068, -38.96708698840265, -97.20583909392128, -108.13830998541118, -52.80402044101346, -37.54617304003958, -56.412965001424084, -31.169353812596572, -44.08450300793773, -45.134477835003665, -69.26140006570098, -46.283561147182645, -70.0046510716963, -62.388694451570146, -63.16160598245163, -90.68758033831841, -105.24645097016509, -46.820862074910394, -103.49308090489517, -84.53721988060632, -73.23261908159539, -54.60909629363111, -90.21906967584808, -126.97926503267226, -84.69837947273152, -169.00590339924707, -56.94185881252751, -42.04519362764758, -75.86888008226839, -65.55969200450375, -56.585585529120436, -179.87937636314078, -85.92046201832395, -48.248692437229934, -32.58967189828772, -104.20680145457798, -46.28338106461178, -41.028126704179215, -104.60950902334417, -57.65439735274087, -137.96796719117563, -79.89435247456827, -57.31516024135204, -110.43292286058532, -120.69982696675712, -58.38735168812701, -78.94068628501785, -48.206477700174304, -44.27070124593095, -42.37894025412056, -155.54838903790912, -74.23947351255752, -39.23248738903292, -65.34165944419972, -87.13067461102179, -74.12655975052522, -94.80998284125587, -129.1927408159765, -60.0864565034926, -44.69685572756444, -135.54463096863586, -117.57020514609823, -74.86567741130766, -62.51221040723401, -112.42303635361225, -113.02642767120246, -97.46967244771588, -151.82485543938185, -59.706242523569756, -89.0587563424232, -88.60717365233512, -48.374503177966204, -53.41136438413852, -117.88560865564087, -120.46872888546052, -116.59612487332501, -188.62787859598058, -112.54841458084495, -52.03031574898999, -74.4491017555797, -44.39271561969791, -99.9128440716027, -46.376893197956846, -42.84836950148565, -107.47836928685204, -37.831663725778974, -40.758765803453954, -78.26821311356869, -51.9536913932033, -104.93117199538727, -51.11707723663393, -58.05990585671173, -73.24214640035262, -49.313757893151745, -55.883543103590505, -46.01481552400315, -123.41960327766657, -142.74621215768744, -43.43900219872778, -96.16482293451831, -48.33328734529825, -157.6252112682736, -136.82511907310288, -51.748698809493646, -60.85872811802091, -75.97212365361978, -37.209534721472885, -36.276783204827176, -55.7324008598786, -47.74296466825311, -117.50493563087649, -74.72096472231179, -34.02530949929816, -99.63875249687325, -90.94894863894379, -39.569023113517, -50.64130525384698, -182.48065238505058, -73.15762131299708, -138.46441256600357, -50.5553962635706, -124.68609369660011, -138.98114541911184, -48.67077382243727, -118.9911389500484, -55.185917657053864, -85.53451564741152, -101.16511552626741, -45.289890442070806, -51.99087542344608, -87.04625425036544, -90.4423896493892, -80.38720969031961, -94.78745627221569, -49.67397241174214, -53.947414027766406, -97.06162166599508, -62.47262627959323, -49.610048477209745, -76.12315870852804, -95.90397715716867, -101.92669243901751, -47.111003398029496, -94.59281331495207, -99.72618193314868, -83.61927037738016, -59.48375184304852, -53.23803725575029, -112.37485050215705, -74.6477339880566, -45.70525879617925, -57.149455924437035, -55.70829643373309, -181.15136590484295, -69.18728571898836, -51.37866088432856, -79.92758723681668, -39.52199603868661, -58.083804232212536, -48.292112102409135, -44.97958283513867, -40.62192912417968, -135.32101275015705, -54.673150049490204, -46.941980908744604, -60.998112731310215, -83.90157336175976, -150.71962943633736, -37.15450600415751, -125.52930245982681, -96.98534875254359, -61.21496198066396, -38.24836733413158, -59.49959843000655, -63.329296924462085, -172.51414666901005, -46.027712384648396, -107.91539283028386, -51.57367706012236, -42.16096352189363, -143.9900222400294, -96.98972266009707, -81.54472922726939, -44.11304844844035, -113.46104470777223, -51.76264347864306, -159.49579778493788, -95.275955185295, -54.08878693909224, -79.24644006279595, -41.192479616627104, -112.60085357903861, -77.6080640647475, -131.42880678061124, -51.644046110770546, -51.0119194252207, -60.71631192276033, -68.20813773696553, -88.16674805814733, -97.52710285416143, -57.82304617808255, -109.35968455905807, -50.382463592217505, -60.95624354269167, -99.8176724880324, -83.61091647334366, -59.05189527677221, -54.77225884325308, -34.598240447486425, -107.88151624333611, -54.27198929892831, -124.59449349918877, -51.71295486155788, -74.5937103659758, -38.95291068407428, -56.07732623163101, -128.89208736461111, -49.82449600776174, -52.73747706184099, -79.22250769956634, -67.56377462084092, -61.30510181610904, -45.184343414637574, -36.74549588023434, -103.38218705080136, -68.44925314733798, -34.05031584129924, -38.541411837154335, -48.436195349729154, -58.52014219055818, -74.03487858930737, -86.2887261553528, -71.4977321308214, -96.37322525047801, -47.17444272605596, -37.38257049520614, -45.37949974401272, -65.50616675834316, -41.364553451407346, -106.14438437562568, -52.65862568796105, -55.28042283723655, -94.53687054785507, -126.88218562862592, -183.96783743525444, -79.60121616664864, -74.94437067071492, -115.40951693782903, -50.80235419238087, -155.4001703720461, -98.72048046635648, -83.53983646800789, -63.08809203594367, -126.60420985040918, -59.348143693136805, -44.10940776099324, -73.18508146156654, -49.11428543682329, -139.74574500629564, -130.004614435905, -79.41137180598577, -75.30346594138857, -58.93178288289112, -42.60251726933906, -77.65231854912113, -46.37851118759382, -62.83457373338162, -95.45217777978672, -69.78464493031214, -106.23541439165317, -39.92207091417001, -65.53851511774286, -63.58593566104791, -42.460877820446015, -173.76590023086277, -74.55703384985404, -64.23842437621724, -57.29425572242297, -156.84062911458633, -114.46718742974633, -55.06714286010944, -148.4327396932366, -52.47999131454076, -83.5230176305431, -50.50413994676441, -36.12673397039652, -39.99375579031278]
        rewards = [178.01491305038448, 387.2920959372217, 30.9253122487454, 24.621374289061084, 10.287979695572883, 2.6228660696000534, 1.0, 3.8087868951590895, 1.0, 1.0, 27.296139767741227, 55.77459624069565, 2.2984711708304806, 17.40530474628807, 11.343105324654296, 40.94993015691081, 50.08377096287822, 121.34475735612772, 10.043456506999426, 1.0, 1.0, 1.0, 25.008273135304943, 4.204861113257507, 61.21534916884304, 6.948582474828463, 6.92291308590559, 39.45018697915319, 25.69069473133509, 32.63425094692312, 27.334422342514646, 1.0, 326.03889041392483, 2.9730879850029566, 5.0544665752691476, 38.8065811852711, 1.0, 1.0, 2.40116847480547, 4.781705867745488, 76.14297425037857, 7.500211128224185, 5.9393305336336955, 29.71130919057295, 1.0, 1.0, 72.69024504672446, 1.0, 25.797954203433605, 2.3159163934062814, 139.3119373922988, 40.4964462833239, 2.4748427743892947, 26.24525284774729, 34.08086287291175, 107.19216840823785, 3.4535401163232615, 42.041049363620935, 6.287584442945382, 2.5453197092543873, 81.0667605099663, 10.573455344631729, 2.6385410126355144, 75.77200415969621, 6.095527164728448, 6.601111610122572, 1.0, 35.36920992898129, 80.03405481657181, 11.737879013077468, 1.0, 52.31396300100307, 17.539950122357034, 371.6640673334361, 163.03078986738893, 8.01581355841418, 49.366558527438805, 7.791325565456772, 32.236825611850655, 38.744991177171826, 39.60244377885444, 225.8798374191686, 5.034557845995099, 4.917620189704904, 1.0, 249.61728903555422, 179.85453444081247, 77.43175215090041, 53.66814359493973, 30.744395419191505, 13.74023414403965, 115.18394734237246, 1.0, 12.968794400898982, 5.479433157561323, 67.89387301493281, 4.173124560548083, 23.044451933407586, 4.910061674291756, 76.98470995517634, 2.496143253543493, 3.8145450429616736, 39.3926801806674, 3.7607044631986177, 87.46188356134498, 2.1807885061471057, 1.0, 14.317333807871181, 6.830674864344754, 114.1218745271371, 162.52540077960526, 1.0, 13.000317644300274, 151.73705649987699, 8.419130926473727, 1.0, 59.5974162050523, 33.43021352993898, 8.484207940509183, 2.2903977150258967, 5.863209225497448, 39.472914153958506, 7.851488288789523, 3.4026287593976923, 64.89135450813512, 95.96597173945695, 1.0, 14.24036937000621, 13.272675252319399, 2.72785053886905, 239.71341352707637, 2.271350537770074, 68.59997482344198, 20.91633055337719, 8.962168691280045, 55.165876760917584, 32.60335402932132, 1.0, 30.670007142310467, 18.151305806996685, 40.22197760564007, 94.27211536925347, 44.43589699452425, 1.0, 1.0, 12.729683944321035, 38.49907174727073, 2.962150892895492, 57.06426372625836, 47.85232295011904, 31.110067400420203, 16.020847234884602, 10.229778455090113, 4.003056581381884, 6.657795206607183, 96.8747758948388, 10.325569643495914, 6.874144182286344, 368.55649577178195, 71.53729512678959, 15.606466353931882, 194.08541438562688, 2.7421530831006096, 1.0, 21.527088625316317, 438.9062530949026, 1.0, 55.281559443689986, 1.0, 13.910228378903497, 50.26991481540402, 15.246139043593324, 5.995580255251175, 1.0, 1.0, 15.61591158772101, 5.4914560177882255, 41.14759099977355, 4.574231228104394, 29.10803429406524, 2.49563335971017, 4.462926789669779, 21.030188116100675, 3.907612937274818, 323.5784159568084, 21.131814667683376, 1.0, 11.268909190012634, 40.770428219836965, 2.0278364627370227, 1.0, 3.9556842831477734, 1.0, 27.674423861508004, 100.21082218460545, 186.03036770660844, 122.74821154082892, 1.0, 78.7828538931201, 268.61869220828953, 1.0, 1.0, 1.0, 8.04445734805275, 7.49991500275943, 292.787004327608, 3.267162114869543, 105.04522919233293, 1.0, 9.509061337973709, 22.592332642467934, 151.8571280077854, 240.08959033430773, 3.138798880845438, 4.31839241805279, 84.2981882761831, 53.24360478773528, 8.773789573573783, 1.0, 1.0, 1.0, 1.0, 2.3630430398536757, 44.393397532419165, 1.0, 1.0, 5.750529978986852, 37.58271725148649, 27.16470553962235, 45.33563530540708, 2.215004965637846, 19.735945079650463, 4.50785216895784, 24.596705964085274, 2.9716971838187902, 232.666061687055, 209.06398580177665, 39.9532879783453, 87.29623426606639, 29.328187319260984, 1.0, 11.86132135580084, 17.37470903199995, 14.895142418661628, 265.94552215653124, 38.487733183385814, 2.3323309831701717, 1.0, 166.6715883313918, 1.0, 3.855678527399774, 121.6778102476624, 6.6688864036017215, 18.269737045761705, 14.782191369329027, 4.398028132877134, 347.3291207358975, 151.9363738806476, 6.940183425134219, 45.220152490350415, 1.0, 2.1065807283524345, 1.0, 104.74078786790409, 247.84952459097235, 1.0, 19.96757885330522, 6.510253384310765, 2.1000488003909616, 145.78188723849223, 43.617619437604176, 27.465058422553795, 1.0, 219.2305205530258, 23.03783813857728, 65.23150777711936, 132.38848954957604, 54.009716794849936, 14.315033864282952, 8.349837601535041, 57.06513608438638, 1.0, 80.971699812288, 213.99098140037364, 15.723142506362478, 24.411714793654433, 128.0821661764215, 127.32204113963238, 54.05396794634306, 474.3968478861626, 56.58391209834441, 1.0, 25.166922878883184, 15.330503323825882, 29.029233378651746, 2.54757657517502, 1.0, 219.88836335820616, 4.535926165653247, 1.0, 20.558617019387665, 10.399433522496423, 193.7639480333848, 2.751794254419708, 12.065830718601134, 8.268901523829568, 1.0, 36.77322621782006, 1.0, 33.31153906053695, 225.8074850853036, 2.867145712654137, 33.6076376828701, 20.991569452891962, 174.84676473675967, 140.7872950821229, 49.776526328347764, 7.910509376470984, 30.810880493088987, 1.0, 1.0, 8.113135936541692, 2.6599633655318797, 116.38842965223857, 28.517306342119433, 2.9603724860645655, 119.12868894817696, 77.7690098080268, 1.0, 3.172490777422153, 34.997878822110735, 87.35073983142854, 111.8131828735133, 1.0, 112.06255367307774, 189.54043978402666, 4.814275608536809, 77.21930178529738, 25.545256225578044, 41.61733708802806, 77.4153089293247, 2.032892158306419, 1.0, 6.505187075917368, 60.386907730983246, 60.20630013223055, 37.50632487019665, 5.27059823765279, 2.118829447885725, 130.73946109547109, 1.0, 1.0, 6.323715038592589, 45.31736777506701, 102.02381333878174, 1.0, 14.585280766874677, 36.913318574115316, 61.462062330217655, 43.49549293192645, 1.0, 114.80094484658412, 21.75234611541873, 1.0, 15.869694570115758, 28.303699308700956, 125.71763829422952, 2.3734905317325325, 2.6287505779364944, 1.0, 7.7028267524534515, 31.178510379713536, 2.194557370479884, 1.0, 1.0, 209.8106802735279, 1.0, 1.0, 26.52947758639528, 104.9351092928789, 221.31921180875216, 1.0, 112.8680074795426, 137.55728215386068, 1.0, 1.0, 33.29728951028706, 1.0, 163.29617340944165, 26.173399291682845, 42.87520630129394, 1.0, 1.0, 112.25511837982867, 149.43891140896818, 32.41497335269516, 1.0, 3.1761992781466875, 3.727431119310845, 199.96029774234745, 13.208816067867676, 2.1008556984158187, 25.021926519267573, 2.3136963203206706, 48.298521785188896, 22.430736075428044, 257.8181673048306, 11.638886358183177, 8.597406957031957, 1.0, 28.69297285335019, 53.14476016212623, 2.328774737485458, 2.024361571384226, 68.82166633216475, 2.245254896712702, 1.0, 57.28285542515934, 268.93154347516247, 2.172090903229944, 110.90240669706142, 1.0, 23.645525597748602, 13.640385466909851, 61.66983864915075, 2.9421333972001347, 3.5823559398991094, 2.5166182488643356, 2.5667140658113126, 28.13934123940117, 9.63203898667871, 2.3946893986289597, 89.5690869940762, 13.720173499409281, 13.728317117030697, 1.0, 1.0, 123.22749161771546, 96.1067966493263, 1.0, 2.419843265768222, 1.0, 11.837321057614329, 18.118102057833866, 1.0, 3.8971688941395075, 267.2948226392828, 11.017950099200128, 1.0, 1.0, 2.100594303457351, 1.0, 131.6188169550082, 7.921391122284383, 1.0, 212.06558476305995, 198.7897522428067, 246.77910079779352, 11.211877755583908, 12.244373202913708, 95.61002207161069, 2.708549960129753, 143.0648954871874, 12.422635429285775, 25.499847802456777, 18.345222841728873, 109.57251842870409, 31.312333090834933, 1.0, 12.32102926594412, 1.0, 194.83219538422094, 115.02010896735189, 11.52383233513229, 67.28376768816095, 1.0, 3.3774938310958422, 22.052649668491703, 9.577969100748106, 48.46608533210476, 336.982388890965, 2.5784332439422024, 58.259609568921064, 11.738830102136916, 14.778405631548413, 7.0505714681362885, 1.0, 225.19995002364615, 20.207033632182288, 3.5199054647951113, 11.49608618505453, 436.65593534089, 136.33074537851058, 2.38532825470353, 27.71973765720682, 16.900636024473176, 21.038670796165945, 1.0, 1.0, 5.558126170282844]
        # Get a fitted curve for total_rewards progression
        def fit_curve(x, a, b, c):
            return a * x + b * x ** 2 + c


        popt, _ = sp.optimize.curve_fit(fit_curve, CL_CD_list, total_rewards)
        a, b, c = popt
        x_line = np.arange(min(CL_CD_list), max(CL_CD_list), 0.1)
        y_line = fit_curve(x_line, a, b, c)
        y_line = fit_curve(x_line, a, b, c)

        # Plotting of the rewards progression vs CL_CD
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.scatter(CL_CD_list, total_rewards, s=3, color='#4292c6', label='Total rewards')
        plt.scatter(CL_CD_list, penalties, s=3, color='#ef6548', label='Penalties')
        plt.scatter(CL_CD_list, rewards, s=3, color="#78c679", label='Gains')
        plt.plot(x_line, y_line, color="#045a8d", label='Fitted total rewards')
        plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color="#737373", linestyles='dashed')
        plt.xlabel('Objective value')
        plt.ylabel('Reward value')
        fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs Objective reached for {aircraft_config}, 500 000 training steps')
        plt.show()
    if ANALYSIS[6] == 4:
        aircraft_config = 'DA_50'
        res = Results(aircraft_config)
        CL_CD_list = res.objectives
        total_rewards = res.total_rewards
        penalties = res.penalties
        rewards = res.rewards


        # Get a fitted curve for total_rewards progression
        def fit_curve(x, a, b, c):
            return a * x + b * x ** 2 + c


        popt, _ = sp.optimize.curve_fit(fit_curve, CL_CD_list, total_rewards)
        a, b, c = popt
        x_line = np.arange(min(CL_CD_list), max(CL_CD_list), 0.1)
        y_line = fit_curve(x_line, a, b, c)
        y_line = fit_curve(x_line, a, b, c)

        # Plotting of the rewards progression vs CL_CD
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.scatter(CL_CD_list, total_rewards, s= 3,color='#4292c6', label='Total rewards')
        plt.scatter(CL_CD_list, penalties, s= 3,color='#ef6548', label='Penalties')
        plt.scatter(CL_CD_list, rewards, s= 3,color="#78c679", label='Gains')
        plt.plot(x_line, y_line, color="#045a8d", label='Fitted total rewards')
        plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color="#737373", linestyles='dashed')
        plt.xlabel('Objective value')
        plt.ylabel('Reward value')
        fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs Objective reached for {aircraft_config}, 500 000 training steps')
        plt.show()
    if ANALYSIS[6] == 5:
        aircraft_config = 'ASK_21'
        res = Results(aircraft_config)
        CL_CD_list = res.objectives
        total_rewards = res.total_rewards
        penalties = res.penalties
        rewards = res.rewards


        # Get a fitted curve for total_rewards progression
        def fit_curve(x, a, b, c):
            return a * x + b * x ** 2 + c


        popt, _ = sp.optimize.curve_fit(fit_curve, CL_CD_list, total_rewards)
        a, b, c = popt
        x_line = np.arange(min(CL_CD_list), max(CL_CD_list), 0.1)
        y_line = fit_curve(x_line, a, b, c)
        y_line = fit_curve(x_line, a, b, c)

        # Plotting of the rewards progression vs CL_CD
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.scatter(CL_CD_list, total_rewards, s= 3,color='#4292c6', label='Total rewards')
        plt.scatter(CL_CD_list, penalties, s= 3,color='#ef6548', label='Penalties')
        plt.scatter(CL_CD_list, rewards, s= 3,color="#78c679", label='Gains')
        plt.plot(x_line, y_line, color="#045a8d", label='Fitted total rewards')
        plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color="#737373", linestyles='dashed')
        plt.xlabel('Objective value')
        plt.ylabel('Reward value')
        fig.gca().legend(loc="lower left")
        # plt.title(f'Rewards vs Objective reached for {aircraft_config}, 500 000 training steps')
        plt.show()
    if ANALYSIS[6] == 6:
        aircraft_config = 'F_50'
        res = Results(aircraft_config)
        CL_CD_list = res.objectives
        total_rewards = res.total_rewards
        penalties = res.penalties
        rewards = res.rewards


        # Get a fitted curve for total_rewards progression
        def fit_curve(x, a, b, c):
            return a * x + b * x ** 2 + c


        popt, _ = sp.optimize.curve_fit(fit_curve, CL_CD_list, total_rewards)
        a, b, c = popt
        x_line = np.arange(min(CL_CD_list), max(CL_CD_list), 0.1)
        y_line = fit_curve(x_line, a, b, c)
        y_line = fit_curve(x_line, a, b, c)

        # Plotting of the rewards progression vs CL_CD
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.scatter(CL_CD_list, total_rewards, s= 3,color='#4292c6', label='Total rewards')
        plt.scatter(CL_CD_list, penalties, s= 3,color='#ef6548', label='Penalties')
        plt.scatter(CL_CD_list, rewards, s= 3,color="#78c679", label='Gains')
        plt.plot(x_line, y_line, color="#045a8d", label='Fitted total rewards')
        plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color="#737373", linestyles='dashed')
        plt.xlabel('Objective value')
        plt.ylabel('Reward value')
        fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs Objective reached for {aircraft_config}, 500 000 training steps')
        plt.show()
    #### ANALYSIS 9: Optimization progression vs timestep
    # Show all rewards vs timestep as it progresses
    #
    if ANALYSIS[7] == 1:
        rewards_raw = [1.2327992638120064, 2.8481938947443064, 4.743747283966832, 6.700463593754543, 8.849684467066965,
                       11.21049088568803, 14.18879916063459, 14.18879916063459, 14.18879916063459, 17.51118706474888,
                       21.39337190390812, 21.39337190390812, 21.39337190390812, 21.39337190390812, 25.277910747969003,
                       29.203800205727752, 33.76985313941353, 38.53886970737511, 43.71344318069408, 49.26688674251248,
                       55.469378018657515, 55.469378018657515, 61.93894738933346, 68.80180920463776, 76.6871719484436,
                       85.64508122918231, 96.94004343610752, 108.48305297462937, 120.82415189088289, 133.50560036937372,
                       133.50560036937372, 146.73829600534125, 160.7653375250964, 175.34169351634662, 191.0065796482105,
                       206.92455079178634, 206.92455079178634, 224.75940081263838, 246.7203072913966,
                       269.84520728490776, 294.06172436372844, 320.32466881902155, 346.6966889987715, 346.6966889987715,
                       374.5243672039849, 404.07699658047886, 404.07699658047886, 404.07699658047886,
                       404.07699658047886, 404.07699658047886, 404.07699658047886, 404.07699658047886,
                       404.07699658047886, 404.07699658047886, 404.07699658047886, 404.07699658047886,
                       404.07699658047886, 404.07699658047886, 404.07699658047886, 404.07699658047886,
                       404.07699658047886, 404.07699658047886, 404.07699658047886, 404.07699658047886,
                       404.07699658047886, 404.07699658047886, 404.07699658047886, 404.07699658047886,
                       404.07699658047886, 404.07699658047886, 642.2321441656009]
        penalties_raw = [0, 0, 0, 0, 0, 0, 0, -0.4155052714686423, -0.516109424656276, -0.516109424656276,
                         -0.516109424656276, -0.9342482701222325, -1.0363253001238322, -1.136925608527437,
                         -1.136925608527437, -1.136925608527437, -1.136925608527437, -1.136925608527437,
                         -1.136925608527437, -1.136925608527437, -1.136925608527437, -1.5491911052829055,
                         -1.5491911052829055, -1.5491911052829055, -1.5491911052829055, -1.5491911052829055,
                         -1.5491911052829055, -1.5491911052829055, -1.5491911052829055, -1.5491911052829055,
                         -1.9537587624118344, -1.9537587624118344, -1.9537587624118344, -1.9537587624118344,
                         -1.9537587624118344, -1.9537587624118344, -2.3539686874276993, -2.3539686874276993,
                         -2.3539686874276993, -2.3539686874276993, -2.3539686874276993, -2.3539686874276993,
                         -2.3539686874276993, -3.5563126409122523, -3.5563126409122523, -3.5563126409122523,
                         -4.806288162229519, -6.16421077618192, -8.185189990845597, -10.226957948925586,
                         -12.325567577018365, -14.502476895313855, -16.575152437023153, -18.832680289604266,
                         -21.092775832512377, -23.369368756878266, -25.74121647550635, -27.914804888268883,
                         -30.18366130788625, -32.42897021675024, -34.85299462134954, -37.15654483494583,
                         -39.65363693309354, -41.98597974936946, -44.24786606855139, -46.477223677981925,
                         -48.75928021423588, -51.20213411537179, -53.89611107348251, -56.520331519384776,
                         -59.169363386774776]
        total_reward_raw = [1.2327992638120064, 2.8481938947443064, 4.743747283966832, 6.700463593754543,
                            8.849684467066965, 11.21049088568803, 14.18879916063459, 13.773293889165947,
                            13.672689735978313, 16.995077640092607, 20.877262479251844, 20.459123633785886,
                            20.357046603784287, 20.25644629538068, 24.14098513944156, 28.06687459720031,
                            32.63292753088609, 37.40194409884768, 42.576517572166644, 48.12996113398505,
                            54.33245241013008, 53.920186913374614, 60.389756284050556, 67.25261809935486,
                            75.1379808431607, 84.09589012389941, 95.39085233082463, 106.93386186934649, 119.2749607856,
                            131.95640926409084, 131.55184160696191, 144.78453724292945, 158.81157876268458,
                            173.3879347539348, 189.05282088579867, 204.9707920293745, 204.57058210435864,
                            222.40543212521067, 244.3663386039689, 267.49123859748005, 291.70775567630074,
                            317.97070013159384, 344.3427203113438, 343.1403763578592, 370.96805456307266,
                            400.52068393956654, 399.2707084182493, 397.9127858042969, 395.89180658963323,
                            393.8500386315532, 391.75142900346043, 389.57451968516494, 387.5018441434556,
                            385.2443162908745, 382.9842207479664, 380.7076278236005, 378.3357801049724,
                            376.1621916922099, 373.89333527259254, 371.64802636372855, 369.2240019591292,
                            366.92045174553294, 364.4233596473852, 362.0910168311093, 359.82913051192736,
                            357.59977290249685, 355.3177163662429, 352.874862465107, 350.1808855069963,
                            347.556665061094, 583.062780778826]
        CL_CD_list_raw = [28.181443854853704, 28.323378218539382, 28.413583890447097, 28.432075010484535,
                          28.48787176682434, 28.545478895945386, 28.696026144974798, 28.696026144974798,
                          28.696026144974798, 28.770999322134298, 28.8826113322492, 28.8826113322492, 28.8826113322492,
                          28.8826113322492, 28.88305724304192, 28.890860935828872, 29.005239189056976,
                          29.039252094822256, 29.104428346982587, 29.162305908263757, 29.25562055298055,
                          29.25562055298055, 29.2921274585214, 29.344100551378265, 29.470522860288565,
                          29.59187785610762, 29.826131633073068, 29.8490247473674, 29.920529437770632,
                          29.95008584078727, 29.95008584078727, 29.996854763417748, 30.062017008371537,
                          30.105652326075308, 30.18898413577904, 30.207801978261816, 30.207801978261816,
                          30.34423312487862, 30.607731590365688, 30.676003722982266, 30.73797894752584,
                          30.849318970690344, 30.85508855793274, 30.85508855793274, 30.93060682074285,
                          31.01676042442508, 31.01676042442508, 31.01676042442508, 31.01676042442508, 31.01676042442508,
                          31.01676042442508, 31.01676042442508, 31.01676042442508, 31.01676042442508, 31.01676042442508,
                          31.01676042442508, 31.01676042442508, 31.01676042442508, 31.01676042442508, 31.01676042442508,
                          31.01676042442508, 31.01676042442508, 31.01676042442508, 31.01676042442508, 31.01676042442508,
                          31.01676042442508, 31.01676042442508, 31.01676042442508, 31.01676042442508, 31.01676042442508,
                          31.01676042442508]

        # volume_constraint = [1.9096999631787868, 1.9148096219353101, 1.9266204025412366, 1.9503609231632695,
        #                      1.9671118313577831, 1.9824023755047486, 1.9705988838058093, 2.016167943349166,
        #                      2.0458348277586005, 2.0472281606437983, 2.0492607857880136, 2.0880877423913855,
        #                      2.1094944583812674, 2.1248020472275853, 2.138282573430417, 2.1511098948509657,
        #                      2.1437903142650825, 2.1620796039708, 2.1782652924510457, 2.1929606322904025,
        #                      2.193568204868291, 2.2138116843509357, 2.217505005917481, 2.221144350169956,
        #                      2.1991517571038117, 2.1947907859709743, 2.143611702061119, 2.1513904733630573,
        #                      2.148173018225596, 2.1548594139423356, 2.1751362699737338, 2.1779231582475536,
        #                      2.174679525777191, 2.1667428520809078, 2.161939621916961, 2.165656496473359,
        #                      2.174233163207396, 2.14829981867362, 2.0918323653373943, 2.072726100164461,
        #                      2.0594383022532305, 2.03732782105529, 2.0298747263223524, 2.010323707176524,
        #                      2.0254460656549966, 2.0148293913270408, 2.002444532340909, 1.981308698268271,
        #                      1.9515208026142181, 1.9484235995507642, 1.9388863350691063, 1.9252530200062745,
        #                      1.9426445026850554, 1.9124030127813794, 1.9119561167965458, 1.9089666679104649,
        #                      1.892970158623698, 1.924891457909035, 1.9104898772842134, 1.9132251799961704,
        #                      1.8851730638184732, 1.9041084656788358, 1.8751042113563696, 1.9007202961021425,
        #                      1.9116823055203702, 1.9172784175041824, 1.9091229491772288, 1.8834921206534716,
        #                      1.8468803795437556, 1.857256954835243, 1.8538788830323167]
        # volume_product_constraint = [1.90256528941378, 1.9047735942809436, 1.9073745127706863, 1.9101066729289264,
        #                              1.9127848816871378, 1.9155084358110321, 1.9180075479314325, 1.9209867323096983,
        #                              1.9238601593882705, 1.9265472786988016, 1.9293120480054926, 1.9323396040774945,
        #                              1.9351909875398952, 1.9380635005359894, 1.9408398239981568, 1.9437877571490318,
        #                              1.9464812935578368, 1.9495028291891152, 1.952538662598192, 1.9555798583288124,
        #                              1.958574420780178, 1.961729944511261, 1.9647655626505762, 1.9677465632528772,
        #                              1.9690832730701755, 1.972181440700551, 1.9735496366567211, 1.976780035960429,
        #                              1.9799687998999318, 1.983252117731658, 1.9867220987107699, 1.9899395195164638,
        #                              1.992526051760648, 1.9946977745535877, 1.997299041440028, 2.0004918186372453,
        #                              2.0029535523920328, 2.006103565000144, 2.0079020378327064, 2.0076367200064538,
        #                              2.0086664214521157, 2.0094555208066684, 2.009963913416289, 2.0107947522477745,
        #                              2.0114155642404814, 2.0119222201212983, 2.012378895182933, 2.011917253703239,
        #                              2.011602791040574, 2.012147123929979, 2.0123902038953076, 2.0118770332376337,
        #                              2.0116907039087764, 2.012212603784928, 2.0121778022753043, 2.011820812051885,
        #                              2.0108030160948527, 2.0109246458492045, 2.0121179242652145, 2.01100192125707,
        #                              2.0110748345138916, 2.0112321745617203, 2.0121547419353467, 2.0124651277552594,
        #                              2.0121931666007717, 2.0125408257084207, 2.0129077858705027, 2.012351830964208,
        #                              2.012765459986507, 2.0131394704112635, 2.013368826516397]
        # volume_plot  = [volume_product_constraint[i] - volume_constraint[i]  for i in range(len(volume_constraint)) ]
        max_scale = max(total_reward_raw)
        max_clcd = max(CL_CD_list_raw)
        rewards = [i / max_scale for i in rewards_raw]
        penalties = [i / max_scale for i in penalties_raw]
        total_reward = [i / max_scale for i in total_reward_raw]
        CL_CD_list = [(i - AC.objective_start) / (max_clcd - AC.objective_start) for i in CL_CD_list_raw]

        # for i in range(len(clcd)):
        #     clcd_mul.append(clcd[i] * 20)
        #
        step = np.linspace(0, len(CL_CD_list), len(CL_CD_list))
        # print(len(step), len(clcd))


        # plt.scatter(CL_CD_list, total_rewards, s= 3,color='#4292c6', label='Total rewards')
        # plt.scatter(CL_CD_list, penalties, s= 3,color='#ef6548', label='Penalties')
        # plt.scatter(CL_CD_list, rewards, s= 3,color="#78c679", label='Gains')
        # plt.plot(x_line, y_line, color="#045a8d", label='Fitted total rewards')
        # plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color="#737373", linestyles='dashed')

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.plot(step, total_reward, color='#4292c6', label='Total rewards')
        plt.plot(step, penalties, color='#ef6548', label='Penalties')
        plt.plot(step, rewards, color="#78c679", label='Gains')
        # plt.plot(step, volume_plot, color='grey', label='volume constraint')
        plt.plot(step, CL_CD_list, color="#807dba", label='Objective value')
        plt.hlines(0, 0, len(CL_CD_list), color="#737373", linestyles='dashed')
        plt.vlines(45.5, -0.1, 1.1, colors="#636363", linestyles='dashed')
        plt.xlabel('Step')
        fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs step')


        # time_step = 1
        # fig = plt.figure()
        # plt.scatter(time_step, total_reward, color='black', label='Total rewards')
        # plt.scatter(CL_CD_list, penalties, color='red', label='Penalties')
        # plt.scatter(CL_CD_list, rewards, color='green', label='Rewards')
        # plt.plot(x_line, y_line, color='blue', label='Total rewards fit')
        # plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color='orange')
        # fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs Objective reached for {aircraft_config}, 500 000 training steps')
        plt.show()
    if ANALYSIS[7] == 2:

        CL_CD_list_raw = [28.16016576893374, 28.30556603033335, 28.30556603033335, 28.30556603033335, 28.30556603033335, 28.30556603033335, 28.30556603033335, 28.30556603033335, 28.30556603033335, 28.42685436269166, 28.42685436269166, 28.42685436269166, 28.42685436269166, 28.541751535934665, 28.63873335292137, 28.63873335292137, 28.63873335292137, 28.63873335292137, 28.63873335292137, 28.63873335292137, 28.63873335292137, 28.63873335292137, 28.87726350505615, 28.87726350505615, 28.87726350505615, 28.87726350505615, 28.87726350505615, 29.056233173218082, 29.217854384275366, 29.298837751343637, 29.41190288540872, 29.41190288540872, 29.698430651254526, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.733128424420503, 29.99458294458862, 29.99458294458862, 30.286946674996184, 30.286946674996184, 30.286946674996184, 30.448422856632114, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576, 30.555059646189576]
        total_reward_raw = [1.1812611171307288, 2.744821166389722, 2.3203414095343367, 1.8750048185636117, 1.3818884997827032, 0.8401459582021329, 0.7201230253583021, 0.6094992060005139, 0.5058840505246442, 2.44520078312803, 2.0204793246846, 1.5954048985319011, 1.4948901861163957, 3.841603839277857, 6.57361572388403, 6.167198601667768, 5.754283887472474, 5.270447228023907, 5.16336012907839, 5.059706300126673, 4.956864411564018, 4.854035689776095, 8.70806290355706, 8.306692633377816, 7.898261408705454, 7.477003723149559, 7.054882037764098, 11.927441946806313, 17.86148480788676, 24.380964765678144, 31.780436388079814, 31.36883748042659, 41.345096381341754, 51.66889978601726, 51.233669519530245, 50.77405170024059, 50.66671612018207, 50.182620119065305, 49.63487937877788, 49.05502265420478, 48.72989593732257, 48.407155763231806, 47.604366064279844, 47.281760266871466, 46.494890080497974, 46.17714689021909, 45.86981958190292, 45.56698821896954, 45.26551759723626, 58.47107408248024, 57.8581175268646, 74.87037322290682, 74.2379078960371, 73.9375530411643, 93.33567918590582, 114.4257505502484, 112.42082524954445, 110.63065672440499, 107.84809496200543, 104.87472138902561, 101.27426988249645, 97.25858016799256, 93.11540367287904, 88.77927760718771, 84.07368418361696, 79.18080466291948, 74.7752116035667, 70.19813609538882, 65.3003924687825, 60.13047488469925, 54.948153395658565, 49.35195463553321, 42.9937113740431, 35.94043724199984, 28.945713474396296, 22.13348533806817, 15.346315694353162, 9.107199200948294, 3.155705689799662, -2.2934886123093996, 136.56680441093033]
        penalties_raw = [0, 0, -0.42447975685538547, -0.8698163478261105, -1.3629326666070192, -1.9046752081875895, -2.02469814103142, -2.1353219603892084, -2.238937115865078, -2.238937115865078, -2.6636585743085077, -3.0887330004612066, -3.1892477128767123, -3.1892477128767123, -3.1892477128767123, -3.5956648350929745, -4.008579549288268, -4.492416208736835, -4.599503307682353, -4.703157136634069, -4.805999025196725, -4.908827746984648, -4.908827746984648, -5.310198017163891, -5.718629241836253, -6.139886927392149, -6.56200861277761, -6.56200861277761, -6.56200861277761, -6.56200861277761, -6.56200861277761, -6.973607520430833, -6.973607520430833, -6.973607520430833, -7.408837786917854, -7.868455606207512, -7.975791186266028, -8.459887187382797, -9.007627927670223, -9.587484652243326, -9.912611369125536, -10.235351543216296, -11.038141242168255, -11.36074703957663, -12.147617225950121, -12.465360416229005, -12.77268772454518, -13.075519087478558, -13.376989709211836, -13.376989709211836, -13.989946264827477, -13.989946264827477, -14.622411591697196, -14.922766446569986, -14.922766446569986, -14.922766446569986, -16.927691747273933, -18.7178602724134, -21.500422034812956, -24.473795607792777, -28.074247114321928, -32.08993682882582, -36.233113323939335, -40.569239389630674, -45.27483281320142, -50.16771233389889, -54.57330539325168, -59.150380901429564, -64.04812452803588, -69.21804211211915, -74.40036360115982, -79.99656236128519, -86.3548056227753, -93.40807975481857, -100.40280352242212, -107.21503165875025, -114.00220130246525, -120.24131779587012, -126.19281130701874, -131.6420056091278, -137.43238599945784]
        rewards_raw = [1.1812611171307288, 2.744821166389722, 2.744821166389722, 2.744821166389722, 2.744821166389722, 2.744821166389722, 2.744821166389722, 2.744821166389722, 2.744821166389722, 4.684137898993107, 4.684137898993107, 4.684137898993107, 4.684137898993107, 7.0308515521545685, 9.76286343676074, 9.76286343676074, 9.76286343676074, 9.76286343676074, 9.76286343676074, 9.76286343676074, 9.76286343676074, 9.76286343676074, 13.616890650541706, 13.616890650541706, 13.616890650541706, 13.616890650541706, 13.616890650541706, 18.489450559583922, 24.42349342066437, 30.942973378455754, 38.342445000857424, 38.342445000857424, 48.31870390177259, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 58.642507306448096, 71.84806379169208, 71.84806379169208, 88.8603194877343, 88.8603194877343, 88.8603194877343, 108.25844563247581, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 129.34851699681838, 273.9991904103881]






        # volume_constraint = [1.9096999631787868, 1.9148096219353101, 1.9266204025412366, 1.9503609231632695,
        #                      1.9671118313577831, 1.9824023755047486, 1.9705988838058093, 2.016167943349166,
        #                      2.0458348277586005, 2.0472281606437983, 2.0492607857880136, 2.0880877423913855,
        #                      2.1094944583812674, 2.1248020472275853, 2.138282573430417, 2.1511098948509657,
        #                      2.1437903142650825, 2.1620796039708, 2.1782652924510457, 2.1929606322904025,
        #                      2.193568204868291, 2.2138116843509357, 2.217505005917481, 2.221144350169956,
        #                      2.1991517571038117, 2.1947907859709743, 2.143611702061119, 2.1513904733630573,
        #                      2.148173018225596, 2.1548594139423356, 2.1751362699737338, 2.1779231582475536,
        #                      2.174679525777191, 2.1667428520809078, 2.161939621916961, 2.165656496473359,
        #                      2.174233163207396, 2.14829981867362, 2.0918323653373943, 2.072726100164461,
        #                      2.0594383022532305, 2.03732782105529, 2.0298747263223524, 2.010323707176524,
        #                      2.0254460656549966, 2.0148293913270408, 2.002444532340909, 1.981308698268271,
        #                      1.9515208026142181, 1.9484235995507642, 1.9388863350691063, 1.9252530200062745,
        #                      1.9426445026850554, 1.9124030127813794, 1.9119561167965458, 1.9089666679104649,
        #                      1.892970158623698, 1.924891457909035, 1.9104898772842134, 1.9132251799961704,
        #                      1.8851730638184732, 1.9041084656788358, 1.8751042113563696, 1.9007202961021425,
        #                      1.9116823055203702, 1.9172784175041824, 1.9091229491772288, 1.8834921206534716,
        #                      1.8468803795437556, 1.857256954835243, 1.8538788830323167]
        # volume_product_constraint = [1.90256528941378, 1.9047735942809436, 1.9073745127706863, 1.9101066729289264,
        #                              1.9127848816871378, 1.9155084358110321, 1.9180075479314325, 1.9209867323096983,
        #                              1.9238601593882705, 1.9265472786988016, 1.9293120480054926, 1.9323396040774945,
        #                              1.9351909875398952, 1.9380635005359894, 1.9408398239981568, 1.9437877571490318,
        #                              1.9464812935578368, 1.9495028291891152, 1.952538662598192, 1.9555798583288124,
        #                              1.958574420780178, 1.961729944511261, 1.9647655626505762, 1.9677465632528772,
        #                              1.9690832730701755, 1.972181440700551, 1.9735496366567211, 1.976780035960429,
        #                              1.9799687998999318, 1.983252117731658, 1.9867220987107699, 1.9899395195164638,
        #                              1.992526051760648, 1.9946977745535877, 1.997299041440028, 2.0004918186372453,
        #                              2.0029535523920328, 2.006103565000144, 2.0079020378327064, 2.0076367200064538,
        #                              2.0086664214521157, 2.0094555208066684, 2.009963913416289, 2.0107947522477745,
        #                              2.0114155642404814, 2.0119222201212983, 2.012378895182933, 2.011917253703239,
        #                              2.011602791040574, 2.012147123929979, 2.0123902038953076, 2.0118770332376337,
        #                              2.0116907039087764, 2.012212603784928, 2.0121778022753043, 2.011820812051885,
        #                              2.0108030160948527, 2.0109246458492045, 2.0121179242652145, 2.01100192125707,
        #                              2.0110748345138916, 2.0112321745617203, 2.0121547419353467, 2.0124651277552594,
        #                              2.0121931666007717, 2.0125408257084207, 2.0129077858705027, 2.012351830964208,
        #                              2.012765459986507, 2.0131394704112635, 2.013368826516397]
        # volume_plot  = [volume_product_constraint[i] - volume_constraint[i]  for i in range(len(volume_constraint)) ]
        max_scale = max(total_reward_raw)
        max_clcd = max(CL_CD_list_raw)
        rewards = [i / max_scale for i in rewards_raw]
        penalties = [i / max_scale for i in penalties_raw]
        total_reward = [i / max_scale for i in total_reward_raw]
        CL_CD_list = [(i - AC.objective_start) / (max_clcd - AC.objective_start) for i in CL_CD_list_raw]

        # for i in range(len(clcd)):
        #     clcd_mul.append(clcd[i] * 20)
        #
        step = np.linspace(0, len(CL_CD_list), len(CL_CD_list))
        # print(len(step), len(clcd))


        # plt.scatter(CL_CD_list, total_rewards, s= 3,color='#4292c6', label='Total rewards')
        # plt.scatter(CL_CD_list, penalties, s= 3,color='#ef6548', label='Penalties')
        # plt.scatter(CL_CD_list, rewards, s= 3,color="#78c679", label='Gains')
        # plt.plot(x_line, y_line, color="#045a8d", label='Fitted total rewards')
        # plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color="#737373", linestyles='dashed')

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.plot(step, total_reward, color='#4292c6', label='Total rewards')
        plt.plot(step, penalties, color='#ef6548', label='Penalties')
        plt.plot(step, rewards, color="#78c679", label='Gains')
        # plt.plot(step, volume_plot, color='grey', label='volume constraint')
        plt.plot(step, CL_CD_list, color="#807dba", label='Objective value')
        plt.hlines(0, 0, len(CL_CD_list), color="#737373", linestyles='dashed')
        plt.vlines(56, -0.5, 1.5, colors="#636363", linestyles='dashed')
        plt.xlabel('Step')
        fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs step')


        # time_step = 1
        # fig = plt.figure()
        # plt.scatter(time_step, total_reward, color='black', label='Total rewards')
        # plt.scatter(CL_CD_list, penalties, color='red', label='Penalties')
        # plt.scatter(CL_CD_list, rewards, color='green', label='Rewards')
        # plt.plot(x_line, y_line, color='blue', label='Total rewards fit')
        # plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color='orange')
        # fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs Objective reached for {aircraft_config}, 500 000 training steps')
        plt.show()
    if ANALYSIS[7] == 3:
        #at 10 000 timesteps
        CL_CD_list_raw = [28.169200077001076, 28.169200077001076, 28.169200077001076, 28.169200077001076, 28.169200077001076, 28.169200077001076, 28.169200077001076, 28.169200077001076, 28.169200077001076, 28.44542573153243, 28.599292521270346, 28.599292521270346, 28.599292521270346, 28.599292521270346, 28.599292521270346, 28.893309961480416, 28.893309961480416, 28.962246266394672, 28.962246266394672, 28.962246266394672, 28.962246266394672, 28.962246266394672, 28.962246266394672, 28.962246266394672, 28.962246266394672, 28.962246266394672, 28.962246266394672, 28.962246266394672, 28.962246266394672, 28.962246266394672, 28.962246266394672, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.050381541241432, 29.201500164034258, 29.201500164034258, 29.201500164034258, 29.203698882861794, 29.234327057329647, 29.234327057329647, 29.234327057329647, 29.234327057329647, 29.234327057329647, 29.311884075985393, 29.413488935575746, 29.42748303546272, 29.42748303546272, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238, 29.527243386277238]
        total_reward_raw = [1.2029641572391319, 0.7445773466932453, 0.636491030118524, -0.4221106877353691, -1.5875547279522617, -2.8326076725601865, -3.845350858725123, -4.868367792675759, -6.062863553815392, -4.061178748432871, -1.490577552987772, -2.515180429460756, -3.8237728392530013, -5.048644397660493, -6.067633676427446, -2.1287069590581753, -2.5405937920225514, 1.7772577320093448, 1.3627431410353532, 0.9422441108827428, 0.5173699208577126, 0.06520524392486932, -0.6506701509900195, -0.9650988361945961, -1.7374165747867565, -2.052973217772694, -2.8031504929360693, -2.9079162364721554, -3.595194694234296, -4.3270640685424215, -4.631706878750663, 0.20500479149367212, -0.23182617977079123, -0.8964761477618561, -1.441200349019742, -2.1993418503816207, -2.9632925710536777, -3.6210037492179437, -3.9554436592312956, -4.084507358768121, -4.211783862755157, -4.334707109657831, -4.895715800655939, -5.541355175619003, -6.310919171714705, -6.452059846750844, -6.5811996405437005, -6.702652380934231, -7.0115628816466185, -1.1913230631558722, -1.800066534512598, -3.051963412234623, 2.7834910740376104, 8.833651195878005, 7.692641629488481, 6.467356187086123, 5.383084267932291, 4.294787845792236, 10.912038657658151, 18.324386546359165, 25.85099276625407, 24.78607082380355, 33.16139352344349, 31.81759087580284, 30.767571715396603, 29.470115782185847, 28.2008538149042, 26.95825892948051, 25.19720661496843, 23.06157576117373, 20.483171399961073, 18.456397726383994, 15.697829653885341, 13.208609579947446, 9.754156121386316, 6.711049534007117, 3.75050183913999, 0.9309584301704499, -2.1155096991872697, -4.819623947681417, -7.479666273302211, -9.854625650328506, -11.878101325718134, -14.10262277398218, -16.22364787829007, -18.469250822922774, -20.84806664649999, 12.585537600766816]
        penalties_raw = [0, -0.45838681054588654, -0.5664731271206078, -1.625074844974501, -2.7905188851913936, -4.035571829799318, -5.0483150159642545, -6.071331949914891, -7.2658277110545235, -7.2658277110545235, -7.2658277110545235, -8.290430587527506, -9.599022997319752, -10.823894555727245, -11.842883834494199, -11.842883834494199, -12.254770667458574, -12.254770667458574, -12.669285258432566, -13.089784288585177, -13.514658478610206, -13.96682315554305, -14.682698550457939, -14.997127235662516, -15.769444974254677, -16.085001617240614, -16.835178892403988, -16.939944635940073, -17.627223093702213, -18.35909246801034, -18.66373527821858, -18.66373527821858, -19.100566249483045, -19.76521621747411, -20.309940418731994, -21.068081920093874, -21.83203264076593, -22.489743818930197, -22.82418372894355, -22.953247428480374, -23.08052393246741, -23.203447179370084, -23.76445587036819, -24.410095245331256, -25.179659241426958, -25.3207999164631, -25.449939710255954, -25.571392450646485, -25.88030295135887, -25.88030295135887, -26.489046422715596, -27.74094330043762, -27.74094330043762, -27.74094330043762, -28.881952866827145, -30.107238309229505, -31.191510228383336, -32.27980665052339, -32.27980665052339, -32.27980665052339, -32.27980665052339, -33.344728592973915, -33.344728592973915, -34.688531240614566, -35.7385504010208, -37.03600633423156, -38.3052683015132, -39.547863186936894, -41.30891550144897, -43.44454635524367, -46.02295071645633, -48.04972439003341, -50.808292462532066, -53.29751253646996, -56.75196599503109, -59.79507258241029, -62.755620277277416, -65.57516368624695, -68.62163181560467, -71.32574606409882, -73.98578838971962, -76.36074776674592, -78.38422344213555, -80.60874489039959, -82.72976999470748, -84.97537293934019, -87.3541887629174, -89.54759059948937]
        rewards_raw = [1.2029641572391319, 1.2029641572391319, 1.2029641572391319, 1.2029641572391319, 1.2029641572391319, 1.2029641572391319, 1.2029641572391319, 1.2029641572391319, 1.2029641572391319, 3.2046489626216523, 5.7752501580667515, 5.7752501580667515, 5.7752501580667515, 5.7752501580667515, 5.7752501580667515, 9.714176875436022, 9.714176875436022, 14.032028399467919, 14.032028399467919, 14.032028399467919, 14.032028399467919, 14.032028399467919, 14.032028399467919, 14.032028399467919, 14.032028399467919, 14.032028399467919, 14.032028399467919, 14.032028399467919, 14.032028399467919, 14.032028399467919, 14.032028399467919, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 18.868740069712253, 24.688979888203, 24.688979888203, 24.688979888203, 30.524434374475234, 36.574594496315626, 36.574594496315626, 36.574594496315626, 36.574594496315626, 36.574594496315626, 43.19184530818154, 50.604193196882555, 58.130799416777464, 58.130799416777464, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 66.5061221164174, 102.13312820025618]





        # volume_constraint = [1.9096999631787868, 1.9148096219353101, 1.9266204025412366, 1.9503609231632695,
        #                      1.9671118313577831, 1.9824023755047486, 1.9705988838058093, 2.016167943349166,
        #                      2.0458348277586005, 2.0472281606437983, 2.0492607857880136, 2.0880877423913855,
        #                      2.1094944583812674, 2.1248020472275853, 2.138282573430417, 2.1511098948509657,
        #                      2.1437903142650825, 2.1620796039708, 2.1782652924510457, 2.1929606322904025,
        #                      2.193568204868291, 2.2138116843509357, 2.217505005917481, 2.221144350169956,
        #                      2.1991517571038117, 2.1947907859709743, 2.143611702061119, 2.1513904733630573,
        #                      2.148173018225596, 2.1548594139423356, 2.1751362699737338, 2.1779231582475536,
        #                      2.174679525777191, 2.1667428520809078, 2.161939621916961, 2.165656496473359,
        #                      2.174233163207396, 2.14829981867362, 2.0918323653373943, 2.072726100164461,
        #                      2.0594383022532305, 2.03732782105529, 2.0298747263223524, 2.010323707176524,
        #                      2.0254460656549966, 2.0148293913270408, 2.002444532340909, 1.981308698268271,
        #                      1.9515208026142181, 1.9484235995507642, 1.9388863350691063, 1.9252530200062745,
        #                      1.9426445026850554, 1.9124030127813794, 1.9119561167965458, 1.9089666679104649,
        #                      1.892970158623698, 1.924891457909035, 1.9104898772842134, 1.9132251799961704,
        #                      1.8851730638184732, 1.9041084656788358, 1.8751042113563696, 1.9007202961021425,
        #                      1.9116823055203702, 1.9172784175041824, 1.9091229491772288, 1.8834921206534716,
        #                      1.8468803795437556, 1.857256954835243, 1.8538788830323167]
        # volume_product_constraint = [1.90256528941378, 1.9047735942809436, 1.9073745127706863, 1.9101066729289264,
        #                              1.9127848816871378, 1.9155084358110321, 1.9180075479314325, 1.9209867323096983,
        #                              1.9238601593882705, 1.9265472786988016, 1.9293120480054926, 1.9323396040774945,
        #                              1.9351909875398952, 1.9380635005359894, 1.9408398239981568, 1.9437877571490318,
        #                              1.9464812935578368, 1.9495028291891152, 1.952538662598192, 1.9555798583288124,
        #                              1.958574420780178, 1.961729944511261, 1.9647655626505762, 1.9677465632528772,
        #                              1.9690832730701755, 1.972181440700551, 1.9735496366567211, 1.976780035960429,
        #                              1.9799687998999318, 1.983252117731658, 1.9867220987107699, 1.9899395195164638,
        #                              1.992526051760648, 1.9946977745535877, 1.997299041440028, 2.0004918186372453,
        #                              2.0029535523920328, 2.006103565000144, 2.0079020378327064, 2.0076367200064538,
        #                              2.0086664214521157, 2.0094555208066684, 2.009963913416289, 2.0107947522477745,
        #                              2.0114155642404814, 2.0119222201212983, 2.012378895182933, 2.011917253703239,
        #                              2.011602791040574, 2.012147123929979, 2.0123902038953076, 2.0118770332376337,
        #                              2.0116907039087764, 2.012212603784928, 2.0121778022753043, 2.011820812051885,
        #                              2.0108030160948527, 2.0109246458492045, 2.0121179242652145, 2.01100192125707,
        #                              2.0110748345138916, 2.0112321745617203, 2.0121547419353467, 2.0124651277552594,
        #                              2.0121931666007717, 2.0125408257084207, 2.0129077858705027, 2.012351830964208,
        #                              2.012765459986507, 2.0131394704112635, 2.013368826516397]
        # volume_plot  = [volume_product_constraint[i] - volume_constraint[i]  for i in range(len(volume_constraint)) ]
        max_scale = max(total_reward_raw)
        max_clcd = max(CL_CD_list_raw)
        rewards = [i / max_scale for i in rewards_raw]
        penalties = [i / max_scale for i in penalties_raw]
        total_reward = [i / max_scale for i in total_reward_raw]
        CL_CD_list = [(i - AC.objective_start) / (max_clcd - AC.objective_start) for i in CL_CD_list_raw]

        # for i in range(len(clcd)):
        #     clcd_mul.append(clcd[i] * 20)
        #
        step = np.linspace(0, len(CL_CD_list), len(CL_CD_list))
        # print(len(step), len(clcd))


        # plt.scatter(CL_CD_list, total_rewards, s= 3,color='#4292c6', label='Total rewards')
        # plt.scatter(CL_CD_list, penalties, s= 3,color='#ef6548', label='Penalties')
        # plt.scatter(CL_CD_list, rewards, s= 3,color="#78c679", label='Gains')
        # plt.plot(x_line, y_line, color="#045a8d", label='Fitted total rewards')
        # plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color="#737373", linestyles='dashed')

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.plot(step, total_reward, color='#4292c6', label='Total rewards')
        plt.plot(step, penalties, color='#ef6548', label='Penalties')
        plt.plot(step, rewards, color="#78c679", label='Gains')
        # plt.plot(step, volume_plot, color='grey', label='volume constraint')
        plt.plot(step, CL_CD_list, color="#807dba", label='Objective value')
        plt.hlines(0, 0, len(CL_CD_list), color="#737373", linestyles='dashed')
        plt.vlines(63, -2, 3, colors="#636363", linestyles='dashed')
        plt.xlabel('Step')
        fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs step')


        # time_step = 1
        # fig = plt.figure()
        # plt.scatter(time_step, total_reward, color='black', label='Total rewards')
        # plt.scatter(CL_CD_list, penalties, color='red', label='Penalties')
        # plt.scatter(CL_CD_list, rewards, color='green', label='Rewards')
        # plt.plot(x_line, y_line, color='blue', label='Total rewards fit')
        # plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color='orange')
        # fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs Objective reached for {aircraft_config}, 500 000 training steps')
        plt.show()
    if ANALYSIS[7] == 4:
        #OLD reward system with exploited environment
        CL_CD_list_raw = [28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 28.08, 30.222068182167305, 30.222068182167305, 30.378265677037273, 30.378265677037273, 30.378265677037273, 30.391050626283146, 30.629392638700306, 30.629392638700306, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.634964228344995, 30.658750285643922, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782, 30.80231832602782]
        total_reward_raw = [-1.328130605334293, -2.875895681449774, -4.992008685210869, -7.232488440078445, -9.524388953872343, -12.19489401009152, -14.662838483046404, -16.88394959759516, -18.953788087449283, -21.10727323150081, -23.282801171862808, -25.23302571610811, -27.117534019424802, -28.95394742988714, -30.800581777900046, -32.71067397998065, -34.488826122388936, -36.057108360860255, -37.39024425733054, -38.54400273804414, -39.623039541552615, 662.8928705226789, 661.8696710569027, 668.1389486663462, 667.7241773362871, 667.2668662909774, 670.3016514023885, 680.4113388939589, 679.976281846995, 683.2250771905481, 682.7314677258912, 682.2143030447264, 682.1091031870898, 681.6570299382348, 681.1886493653166, 680.6244545233632, 679.7949333383068, 679.4642315083767, 678.6435597482731, 678.3177106461748, 678.2008808843403, 677.884248483503, 677.0271670713764, 676.6997099510208, 675.8616488493532, 674.9389098014414, 673.9620538313317, 673.6206079462493, 673.2792132650768, 672.3351449013423, 671.9874050066287, 671.6443394292736, 670.7273368744798, 670.3873510406077, 669.4896657383091, 668.5692405333878, 667.6056350298613, 667.264389837305, 666.9270717091379, 666.0177022579559, 665.6837559706927, 665.355632552489, 665.0503099518887, 668.6838223786091, 675.7962559749678, 675.1912239892367, 674.573827734711, 672.7483118163583, 670.8367678967182, 668.78757488373, 666.7716508568448, 664.7614900032843, 662.4527288921787, 659.7435794811004, 657.7641049064566, 655.056825330978, 652.9326383057606, 650.2838634911263, 648.1438864871684, 645.5632602435447, 642.8407638359969, 639.7362440917404, 636.0547816703232, 632.3016095589553, 628.6485944656163, 624.9645616231095, 620.6460882947582, 615.8983974296377, 610.2236090632595, 604.7178283160891, 599.0833832471745, 678.3378143705277]
        penalties_raw = [-1.328130605334293, -2.875895681449774, -4.992008685210869, -7.232488440078445, -9.524388953872343, -12.19489401009152, -14.662838483046404, -16.88394959759516, -18.953788087449283, -21.10727323150081, -23.282801171862808, -25.23302571610811, -27.117534019424802, -28.95394742988714, -30.800581777900046, -32.71067397998065, -34.488826122388936, -36.057108360860255, -37.39024425733054, -38.54400273804414, -39.623039541552615, -39.623039541552615, -40.64623900732886, -40.64623900732886, -41.06101033738802, -41.51832138269772, -41.51832138269772, -41.51832138269772, -41.95337842966169, -41.95337842966169, -42.44698789431854, -42.96415257548332, -43.06935243311994, -43.52142568197492, -43.989806254893146, -44.55400109684657, -45.38352228190288, -45.714224111833026, -46.534895871936584, -46.860744974034944, -46.97757473586935, -47.29420713670665, -48.151288548833286, -48.478745669188875, -49.3168067708565, -50.239545818768335, -51.216401788877995, -51.55784767396037, -51.89924235513288, -52.84331071886736, -53.19105061358104, -53.53411619093616, -54.45111874572999, -54.79110457960206, -55.68878988190059, -56.60921508682196, -57.57282059034839, -57.91406578290479, -58.25138391107192, -59.16075336225395, -59.494699649517095, -59.822823067720726, -60.12814566832104, -60.12814566832104, -60.12814566832104, -60.73317765405214, -61.35057390857783, -63.176089826930564, -65.08763374657065, -67.13682675955889, -69.15275078644413, -71.16291164000464, -73.47167275111022, -76.18082216218853, -78.16029673683225, -80.86757631231087, -82.99176333752825, -85.64053815216259, -87.78051515612053, -90.36114139974421, -93.08363780729205, -96.1881575515485, -99.86961997296574, -103.62279208433365, -107.27580717767275, -110.95984002017947, -115.27831334853069, -120.02600421365113, -125.7007925800294, -131.2065733271997, -136.84101839611426, -142.3764294428409]
        rewards_raw = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 702.5159100642315, 702.5159100642315, 708.7851876736751, 708.7851876736751, 708.7851876736751, 711.8199727850862, 721.9296602766566, 721.9296602766566, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 725.1784556202097, 728.81196804693, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 735.9244016432888, 820.7142438133685]




        # volume_constraint = [1.9096999631787868, 1.9148096219353101, 1.9266204025412366, 1.9503609231632695,
        #                      1.9671118313577831, 1.9824023755047486, 1.9705988838058093, 2.016167943349166,
        #                      2.0458348277586005, 2.0472281606437983, 2.0492607857880136, 2.0880877423913855,
        #                      2.1094944583812674, 2.1248020472275853, 2.138282573430417, 2.1511098948509657,
        #                      2.1437903142650825, 2.1620796039708, 2.1782652924510457, 2.1929606322904025,
        #                      2.193568204868291, 2.2138116843509357, 2.217505005917481, 2.221144350169956,
        #                      2.1991517571038117, 2.1947907859709743, 2.143611702061119, 2.1513904733630573,
        #                      2.148173018225596, 2.1548594139423356, 2.1751362699737338, 2.1779231582475536,
        #                      2.174679525777191, 2.1667428520809078, 2.161939621916961, 2.165656496473359,
        #                      2.174233163207396, 2.14829981867362, 2.0918323653373943, 2.072726100164461,
        #                      2.0594383022532305, 2.03732782105529, 2.0298747263223524, 2.010323707176524,
        #                      2.0254460656549966, 2.0148293913270408, 2.002444532340909, 1.981308698268271,
        #                      1.9515208026142181, 1.9484235995507642, 1.9388863350691063, 1.9252530200062745,
        #                      1.9426445026850554, 1.9124030127813794, 1.9119561167965458, 1.9089666679104649,
        #                      1.892970158623698, 1.924891457909035, 1.9104898772842134, 1.9132251799961704,
        #                      1.8851730638184732, 1.9041084656788358, 1.8751042113563696, 1.9007202961021425,
        #                      1.9116823055203702, 1.9172784175041824, 1.9091229491772288, 1.8834921206534716,
        #                      1.8468803795437556, 1.857256954835243, 1.8538788830323167]
        # volume_product_constraint = [1.90256528941378, 1.9047735942809436, 1.9073745127706863, 1.9101066729289264,
        #                              1.9127848816871378, 1.9155084358110321, 1.9180075479314325, 1.9209867323096983,
        #                              1.9238601593882705, 1.9265472786988016, 1.9293120480054926, 1.9323396040774945,
        #                              1.9351909875398952, 1.9380635005359894, 1.9408398239981568, 1.9437877571490318,
        #                              1.9464812935578368, 1.9495028291891152, 1.952538662598192, 1.9555798583288124,
        #                              1.958574420780178, 1.961729944511261, 1.9647655626505762, 1.9677465632528772,
        #                              1.9690832730701755, 1.972181440700551, 1.9735496366567211, 1.976780035960429,
        #                              1.9799687998999318, 1.983252117731658, 1.9867220987107699, 1.9899395195164638,
        #                              1.992526051760648, 1.9946977745535877, 1.997299041440028, 2.0004918186372453,
        #                              2.0029535523920328, 2.006103565000144, 2.0079020378327064, 2.0076367200064538,
        #                              2.0086664214521157, 2.0094555208066684, 2.009963913416289, 2.0107947522477745,
        #                              2.0114155642404814, 2.0119222201212983, 2.012378895182933, 2.011917253703239,
        #                              2.011602791040574, 2.012147123929979, 2.0123902038953076, 2.0118770332376337,
        #                              2.0116907039087764, 2.012212603784928, 2.0121778022753043, 2.011820812051885,
        #                              2.0108030160948527, 2.0109246458492045, 2.0121179242652145, 2.01100192125707,
        #                              2.0110748345138916, 2.0112321745617203, 2.0121547419353467, 2.0124651277552594,
        #                              2.0121931666007717, 2.0125408257084207, 2.0129077858705027, 2.012351830964208,
        #                              2.012765459986507, 2.0131394704112635, 2.013368826516397]
        # volume_plot  = [volume_product_constraint[i] - volume_constraint[i]  for i in range(len(volume_constraint)) ]
        max_scale = max(total_reward_raw)
        max_clcd = max(CL_CD_list_raw)
        rewards = [i / max_scale for i in rewards_raw]
        penalties = [i / max_scale for i in penalties_raw]
        total_reward = [i / max_scale for i in total_reward_raw]
        CL_CD_list = [(i - AC.objective_start) / (max_clcd - AC.objective_start) for i in CL_CD_list_raw]

        # for i in range(len(clcd)):
        #     clcd_mul.append(clcd[i] * 20)
        #
        step = np.linspace(0, len(CL_CD_list), len(CL_CD_list))
        # print(len(step), len(clcd))


        # plt.scatter(CL_CD_list, total_rewards, s= 3,color='#4292c6', label='Total rewards')
        # plt.scatter(CL_CD_list, penalties, s= 3,color='#ef6548', label='Penalties')
        # plt.scatter(CL_CD_list, rewards, s= 3,color="#78c679", label='Gains')
        # plt.plot(x_line, y_line, color="#045a8d", label='Fitted total rewards')
        # plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color="#737373", linestyles='dashed')

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.plot(step, total_reward, color='#4292c6', label='Total rewards')
        plt.plot(step, penalties, color='#ef6548', label='Penalties')
        plt.plot(step, rewards, color="#78c679", label='Gains')
        # plt.plot(step, volume_plot, color='grey', label='volume constraint')
        plt.plot(step, CL_CD_list, color="#807dba", label='Objective value')
        plt.hlines(0, 0, len(CL_CD_list), color="#737373", linestyles='dashed')
        plt.vlines(65, -0.2, 1.2, colors="#636363", linestyles='dashed')
        plt.xlabel('Step')
        fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs step')


        # time_step = 1
        # fig = plt.figure()
        # plt.scatter(time_step, total_reward, color='black', label='Total rewards')
        # plt.scatter(CL_CD_list, penalties, color='red', label='Penalties')
        # plt.scatter(CL_CD_list, rewards, color='green', label='Rewards')
        # plt.plot(x_line, y_line, color='blue', label='Total rewards fit')
        # plt.hlines(0, min(CL_CD_list), max(CL_CD_list), color='orange')
        # fig.gca().legend(loc="upper left")
        # plt.title(f'Rewards vs Objective reached for {aircraft_config}, 500 000 training steps')
        plt.show()
    #### ANALYSIS 10: Constraints progress vs timesteps
    # Show how the constraints progress during the progress of optimization
    # Show behaviour of the ML optimizer and compare with PSO
    # How do they differ in trying to circumvent the constraints
    if ANALYSIS[8] == 1:
        mass_constraint = [14.822811909794462, 14.973632110897354, 15.150583017320226, 15.33567135075339,
                           15.516322722366894, 15.69924384110333, 15.866395336748798, 16.064792111813965,
                           16.255261611188743, 16.432600884871963, 16.614282513663543, 16.812328658157927,
                           16.997992251274276, 17.18419555589481, 17.363371391367433, 17.55277568519557,
                           17.72507753133611, 17.917505991635828, 18.109940619899493, 18.301812591797955,
                           18.489865893604996, 18.68709277461815, 18.875926663301726, 19.060511417798892,
                           19.1430085271227, 19.333571657303246, 19.417441483383506, 19.614773275827126,
                           19.808616166989, 20.007232479820686, 20.21607396811199, 20.408743122725085,
                           20.562959279637592, 20.6919825659004, 20.845974711719705, 21.034168113589597,
                           21.17866139742865, 21.362784288651035, 21.467522431363935, 21.452088593607282,
                           21.51195360135787, 21.557768708424565, 21.587257662537887, 21.63540218657502,
                           21.67133773490489, 21.700640969627262, 21.7270347044741, 21.70035383548226,
                           21.682168844142808, 21.71364160770554, 21.727688070678447, 21.698028406128152,
                           21.687253588416098, 21.71742588069348, 21.715414644868513, 21.694777630574194,
                           21.635880753849655, 21.642923772988816, 21.711953955425543, 21.64739777848016,
                           21.651618760570827, 21.66072569918976, 21.714081893399932, 21.73201655659866,
                           21.716302586959237, 21.736389277122374, 21.75757994445616, 21.72547101215076,
                           21.749362484267014, 21.77095300867752, 21.7841871712988]
        volume_constraint = [1.9096999631787868, 1.9148096219353101, 1.9266204025412366, 1.9503609231632695,
                             1.9671118313577831, 1.9824023755047486, 1.9705988838058093, 2.016167943349166,
                             2.0458348277586005, 2.0472281606437983, 2.0492607857880136, 2.0880877423913855,
                             2.1094944583812674, 2.1248020472275853, 2.138282573430417, 2.1511098948509657,
                             2.1437903142650825, 2.1620796039708, 2.1782652924510457, 2.1929606322904025,
                             2.193568204868291, 2.2138116843509357, 2.217505005917481, 2.221144350169956,
                             2.1991517571038117, 2.1947907859709743, 2.143611702061119, 2.1513904733630573,
                             2.148173018225596, 2.1548594139423356, 2.1751362699737338, 2.1779231582475536,
                             2.174679525777191, 2.1667428520809078, 2.161939621916961, 2.165656496473359,
                             2.174233163207396, 2.14829981867362, 2.0918323653373943, 2.072726100164461,
                             2.0594383022532305, 2.03732782105529, 2.0298747263223524, 2.010323707176524,
                             2.0254460656549966, 2.0148293913270408, 2.002444532340909, 1.981308698268271,
                             1.9515208026142181, 1.9484235995507642, 1.9388863350691063, 1.9252530200062745,
                             1.9426445026850554, 1.9124030127813794, 1.9119561167965458, 1.9089666679104649,
                             1.892970158623698, 1.924891457909035, 1.9104898772842134, 1.9132251799961704,
                             1.8851730638184732, 1.9041084656788358, 1.8751042113563696, 1.9007202961021425,
                             1.9116823055203702, 1.9172784175041824, 1.9091229491772288, 1.8834921206534716,
                             1.8468803795437556, 1.857256954835243, 1.8538788830323167]
        volume_product_constraint = [1.90256528941378, 1.9047735942809436, 1.9073745127706863, 1.9101066729289264,
                                     1.9127848816871378, 1.9155084358110321, 1.9180075479314325, 1.9209867323096983,
                                     1.9238601593882705, 1.9265472786988016, 1.9293120480054926, 1.9323396040774945,
                                     1.9351909875398952, 1.9380635005359894, 1.9408398239981568, 1.9437877571490318,
                                     1.9464812935578368, 1.9495028291891152, 1.952538662598192, 1.9555798583288124,
                                     1.958574420780178, 1.961729944511261, 1.9647655626505762, 1.9677465632528772,
                                     1.9690832730701755, 1.972181440700551, 1.9735496366567211, 1.976780035960429,
                                     1.9799687998999318, 1.983252117731658, 1.9867220987107699, 1.9899395195164638,
                                     1.992526051760648, 1.9946977745535877, 1.997299041440028, 2.0004918186372453,
                                     2.0029535523920328, 2.006103565000144, 2.0079020378327064, 2.0076367200064538,
                                     2.0086664214521157, 2.0094555208066684, 2.009963913416289, 2.0107947522477745,
                                     2.0114155642404814, 2.0119222201212983, 2.012378895182933, 2.011917253703239,
                                     2.011602791040574, 2.012147123929979, 2.0123902038953076, 2.0118770332376337,
                                     2.0116907039087764, 2.012212603784928, 2.0121778022753043, 2.011820812051885,
                                     2.0108030160948527, 2.0109246458492045, 2.0121179242652145, 2.01100192125707,
                                     2.0110748345138916, 2.0112321745617203, 2.0121547419353467, 2.0124651277552594,
                                     2.0121931666007717, 2.0125408257084207, 2.0129077858705027, 2.012351830964208,
                                     2.012765459986507, 2.0131394704112635, 2.013368826516397]
        loading_constraint = [135.77717878582177, 134.59528660237234, 133.107943727121, 131.24174436174457,
                              129.6578376602544, 128.13164548839072, 127.51562562927202, 125.1190249034022,
                              123.26863802832575, 122.30082317597338, 121.28145366836937, 119.27743321748636,
                              117.85978177345241, 116.58430571282892, 115.39167527163217, 114.19926813254749,
                              113.6413662475919, 112.33881434369886, 111.1186409925042, 109.95086941071888,
                              109.14491855607547, 107.89719631156062, 107.0530312727387, 106.2448611074261,
                              106.21486762917786, 105.56876287483935, 106.22897852777187, 105.30425957604191,
                              104.64200637901648, 103.76601888584868, 102.57604916475474, 101.81558334814407,
                              101.26416669360039, 101.06596879062846, 100.55470206444592, 99.8695782180419,
                              99.08793393266761, 98.98406450376952, 99.6963496706096, 100.19563564275657,
                              100.04164225841009, 100.1291333433328, 100.06539181549533, 100.1108188236768,
                              99.54876446299036, 99.54733256504007, 99.60161982059903, 100.18569137557033,
                              100.91708680630056, 100.77073410476463, 100.89068152857487, 101.39779850727327,
                              101.09690288579021, 101.52517928942373, 101.54764952958719, 101.7684600522329,
                              102.53417971850166, 101.8403086368125, 101.70617667250272, 102.13051001798821,
                              102.70674872395041, 102.26436255156092, 102.52976396169558, 101.8944720102448,
                              101.81474282431458, 101.54974196243704, 101.57924218334932, 102.31390942933717,
                              102.95439128635876, 102.58615231224466, 102.57853945818168]

        max_mass = max(mass_constraint)
        max_loading = max(loading_constraint)

        mass_plot = [(i - AC.wing_mass_ub) / max_mass for i in mass_constraint]
        volume_plot = [volume_product_constraint[i] - volume_constraint[i] for i in range(len(volume_constraint))]
        loading_constraint = [(i - AC.wing_loading_ref * 1.1) / (max_loading) for i in loading_constraint]
        step = np.linspace(0, len(mass_constraint), len(mass_constraint))
        # print(len(step), len(clcd))

        # plt.plot(step, total_reward, color='#4292c6', label='Total rewards')
        # plt.plot(step, penalties, color='#ef6548', label='Penalties')
        # plt.plot(step, rewards, color="#78c679", label='Gains')
        # # plt.plot(step, volume_plot, color='grey', label='volume constraint')
        # plt.plot(step, CL_CD_list, color="#807dba", label='Objective value')
        # plt.hlines(0, 0, len(CL_CD_list), color="#737373", linestyles='dashed')
        # plt.xlabel('Objective value')


        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.plot(step, mass_plot, color='#4292c6', label='Mass constraint')
        plt.plot(step, volume_plot, color='#ef6548', label='Volume constraint')
        plt.plot(step, loading_constraint, color="#78c679", label='Wing loading constraint')
        # plt.plot(step, CL_CD_list, color='orange', label='Objective')
        plt.hlines(0, 0, len(mass_constraint), color="#737373",linestyles='dashed')
        plt.vlines(45.5, -0.6, 0.2, colors="#636363", linestyles='dashed')
        fig.gca().legend(loc="upper left")
        plt.xlabel('Step')
        # plt.title(f'Rewards vs step')
        plt.show()
    if ANALYSIS[8] == 2:
        mass_constraint = [14.732106545185744, 14.901177382705777, 15.025586570015005, 15.217266375133425, 15.415626026701643, 15.515586506074461, 15.674559617808686, 15.841886927835965, 15.881947236447925, 16.038330187060517, 16.225848019253757, 16.418966562447423, 16.507781293679198, 16.690967185985794, 16.86649196856172, 17.06372697720145, 17.266932739598733, 17.388600189938014, 17.535633413885655, 17.66870404899899, 17.8441075705394, 17.944036845429874, 18.08618296473764, 18.20672794166892, 18.408658825646835, 18.59602817732769, 18.78059204606533, 18.93480149740682, 19.100572763067504, 19.28646925589558, 19.457162524411252, 19.66649845967632, 19.809301330604324, 19.99398586959977, 20.198642958325166, 20.284611713087504, 20.397621395755756, 20.61837775292083, 20.814188418785687, 20.973409002981278, 21.126135585493703, 21.17843516115558, 21.26467822180184, 21.32196551461198, 21.40432242152192, 21.340752784324984, 21.326341446099722, 21.342755623313856, 21.412765811685304, 21.426457664273507, 21.504266450872862, 21.492932600838767, 21.57685034570121, 21.5836412377488, 21.53106781172152, 21.58658055468718, 21.601737634184413, 21.64560300279039, 21.66282046326341, 21.72279673593699, 21.74411269339005, 21.772487627780663, 21.83400684633928, 21.89620029553697, 21.944738687285266, 21.990610150413296, 22.018403854282088, 22.08814614233374, 22.143663831457385, 22.2153852894837, 22.2069917086164, 22.258302150595757, 22.28893521378096, 22.271245804622726, 22.305736221256637, 22.376373185622384, 22.419988983866407, 22.50485345350096, 22.5142673011441, 22.557283935718097, 22.530093770604196]
        volume_constraint = [1.91600252212913, 1.9196796926126827, 1.9515598177083766, 2.0016556015230567, 2.07083583591288, 2.128059670384493, 2.1533613127409668, 2.1469433359642864, 2.123663879096717, 2.1042416985080807, 2.1563762842097307, 2.185381378299502, 2.1805559952761056, 2.182668076328773, 2.1787388264027845, 2.208004173871675, 2.245502704717443, 2.304560507137713, 2.3007962458807927, 2.3012562256574514, 2.3071908289150933, 2.316742899694601, 2.249486328990959, 2.2620914988631275, 2.2853163364679276, 2.3194594471534034, 2.3322148938430862, 2.2763528131122808, 2.2464674166823846, 2.2410051169209044, 2.2189178139703594, 2.246844123716678, 2.171989235336272, 2.162982357486211, 2.202891817868676, 2.2305343467339833, 2.2249291463482845, 2.2687830092081116, 2.320863843537879, 2.3568392811155645, 2.3479633475855617, 2.336600004837305, 2.384803433379708, 2.3532313008303225, 2.375219499423485, 2.327257640625959, 2.2642750231083637, 2.246898513252809, 2.2529875049532024, 2.1663209880050056, 2.1834644346329912, 2.100567992716236, 2.131933685869476, 2.1118883618675266, 2.0689081382877594, 2.0397918906546346, 1.952989631889934, 1.9059906596358678, 1.8330264016851217, 1.80794744273559, 1.7296638481126374, 1.6826430315010403, 1.6695655161793241, 1.6497906471837505, 1.6123623175521347, 1.5943425884574438, 1.5934739141834608, 1.5777241401739963, 1.5480744518871905, 1.5240171903757052, 1.5227826341203543, 1.4867232115908369, 1.4645777522126522, 1.4069868954604623, 1.412117214158806, 1.3876894217098252, 1.390190793900619, 1.4354201302564722, 1.4595413199483949, 1.5035113124472406, 1.473452937592171]

        volume_product_constraint = [1.90124094610638, 1.9037117373819226, 1.9055361252128629, 1.9083574767944693, 1.911290612945098, 1.912773943824571, 1.9151402171293597, 1.9176404972182421, 1.918240572214211, 1.920588558552635, 1.9234155722107127, 1.9263402810664396, 1.9276898892643344, 1.930482628394862, 1.9331701173286415, 1.936203588148482, 1.939344015988825, 1.9412317106201509, 1.9435203934518253, 1.9455987851120935, 1.9483486482522845, 1.949920531593951, 1.9521630919315502, 1.9540709708660662, 1.957279585300245, 1.9602710599524347, 1.9632312371633274, 1.965714902994516, 1.9683953411811546, 1.9714142906536438, 1.9741986133271012, 1.9776294228341862, 1.9799800983540783, 1.9830326312176014, 1.986431781149462, 1.9878648420074874, 1.9897533719934553, 1.993457991799887, 1.9967612722160106, 1.9994593535246938, 2.0020576260844885, 2.0029496909473345, 2.0044233124219595, 2.005403956999718, 2.0068162483639242, 2.005725868745194, 2.0054789229595213, 2.0057601955961397, 2.0069612065119515, 2.0071963377431095, 2.0085341115793915, 2.0083390826237055, 2.0097844465442822, 2.0099015453092077, 2.008995524223612, 2.0099522357179036, 2.0102136896266645, 2.0109709206769137, 2.011268370853645, 2.012305547384785, 2.0126745495925547, 2.0131660623181253, 2.014232928332124, 2.0153131952357555, 2.016157477134934, 2.016956335836509, 2.017440824774069, 2.0186580668985346, 2.019628599929069, 2.0208844502557546, 2.020737358121871, 2.0216370371741172, 2.022174724731139, 2.0218641792592678, 2.0224698048502927, 2.023711816248259, 2.024479843266598, 2.02597669229945, 2.026142936826897, 2.026903106519232, 2.0264225165755883]

        loading_constraint = [136.0832197029063, 134.95942948388267, 133.76973767340357, 131.18154560970922, 128.20047038540972, 125.89532792007489, 124.67665071721807, 123.95205650572548, 124.12223709152877, 123.88908376858897, 121.79714655456527, 119.99386498362844, 119.6437535955832, 118.53760615424524, 117.7682214812399, 116.05182868827383, 114.17730264428901, 112.69015184072228, 112.05724975680769, 111.37840807567407, 110.60675071160381, 109.77298316459128, 110.69411608746512, 110.21338481794511, 108.82519134169684, 107.43771827183755, 106.50203794835552, 107.09913103721657, 107.16185400499714, 106.487407458893, 106.26559123131727, 104.82043090963674, 105.89067405124695, 105.34423258385154, 103.81516351749734, 103.35023221196678, 102.90697952706354, 101.18542961208398, 99.55365760940428, 98.53297918994079, 98.01188046511717, 97.81062783200575, 96.43425220681044, 96.50842880663208, 95.58578782116552, 96.73358167588887, 97.8515787152961, 98.04731522495076, 97.45891176698025, 98.90978602451408, 98.02969439146399, 99.60486187455653, 98.43848860380494, 98.73637591148479, 99.8901170804387, 100.02118794519866, 101.62213303658487, 102.27285119554845, 103.74546510135107, 103.85478354286523, 105.52546572068375, 106.48571012502303, 106.3617487252628, 106.41446567064625, 107.03483821935191, 107.1711471625707, 106.99668708102931, 106.89060994597826, 107.26862286196847, 107.40879672892406, 107.4899979579732, 108.16371688071092, 108.58589724772555, 110.4711850398184, 110.0512315091316, 110.28720481311751, 109.89183370405492, 107.91621222901414, 107.12428453975465, 105.57193992277458, 106.59757511574334]



        max_mass = max(mass_constraint)
        max_loading = max(loading_constraint)

        mass_plot = [(i - AC.wing_mass_ub) / max_mass for i in mass_constraint]
        volume_plot = [volume_product_constraint[i] - volume_constraint[i] for i in range(len(volume_constraint))]
        loading_constraint = [(i - AC.wing_loading_ref * 1.1) / (max_loading) for i in loading_constraint]
        step = np.linspace(0, len(mass_constraint), len(mass_constraint))
        # print(len(step), len(clcd))

        # plt.plot(step, total_reward, color='#4292c6', label='Total rewards')
        # plt.plot(step, penalties, color='#ef6548', label='Penalties')
        # plt.plot(step, rewards, color="#78c679", label='Gains')
        # # plt.plot(step, volume_plot, color='grey', label='volume constraint')
        # plt.plot(step, CL_CD_list, color="#807dba", label='Objective value')
        # plt.hlines(0, 0, len(CL_CD_list), color="#737373", linestyles='dashed')
        # plt.xlabel('Objective value')


        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.plot(step, mass_plot, color='#4292c6', label='Mass constraint')
        plt.plot(step, volume_plot, color='#ef6548', label='Volume constraint')
        plt.plot(step, loading_constraint, color="#78c679", label='Wing loading constraint')
        # plt.plot(step, CL_CD_list, color='orange', label='Objective')
        plt.hlines(0, 0, len(mass_constraint), color="#737373",linestyles='dashed')
        plt.vlines(56, -0.6, 0.6, colors="#636363", linestyles='dashed')
        fig.gca().legend(loc="upper left")
        plt.xlabel('Step')
        # plt.title(f'Rewards vs step')
        plt.show()
    if ANALYSIS[8] == 3:
        mass_constraint = [14.700563416362995, 14.803514895564543, 14.720212474702912, 14.806030763723854, 14.754550100343728, 14.884395893235233, 15.073758671222972, 15.187644036996248, 15.324986670166158, 15.44885313737879, 15.503463314526606, 15.663517389961854, 15.77089108367512, 15.779157969036941, 15.908182218516597, 15.805769838214271, 15.958712026186305, 16.010826257293747, 16.127860319684608, 16.181059733018113, 16.06506282819766, 16.013397717364306, 15.952796613426182, 15.984023010706295, 16.13518025913304, 16.18676263515079, 16.322776938128502, 16.199857658190588, 16.288632305854755, 16.31684740419134, 16.42242386641339, 16.443425608442933, 16.396280115482426, 16.54101824054233, 16.55339978008493, 16.73668872597611, 16.832634377844602, 16.732528633406147, 16.54383768580511, 16.354591362124854, 16.475217977909402, 16.498199717507646, 16.451713505355666, 16.378611015425246, 16.298713731516422, 16.3557283596439, 16.35063918054035, 16.431202687725964, 16.403133438622884, 16.33854591396663, 16.38344753196235, 16.385573914713436, 16.38250619150026, 16.413614919629445, 16.457245748749887, 16.512559508159523, 16.36145581557129, 16.525977676332726, 16.514123534056505, 16.562735637823486, 16.70450920848339, 16.665627049385865, 16.672047416337595, 16.77380841041144, 16.93558438344645, 16.89007227819029, 16.791359554846107, 16.952092521301118, 16.967034114080224, 16.840838502434185, 16.778335134442358, 16.651722236879888, 16.77673374314013, 16.86558814270499, 16.808462811783464, 16.956903039516835, 16.816200758965508, 16.823829713349664, 16.89060421434471, 17.007223137869392, 17.136069644667593, 17.279103530041656, 17.287225459629408, 17.16616662739517, 17.304407621999434, 17.309119905305817, 17.323887197399124, 17.248024430314832]
        volume_constraint = [1.9014444723628046, 1.95747636084799, 1.9352626564531308, 1.8913308461574407, 1.8713116282446949, 1.8593743937016036, 1.9038225040892491, 1.9035541643316947, 1.8744503613046288, 1.9631490843001065, 1.9327003822804334, 1.9102928047682535, 1.8614050299614426, 1.8757845779607758, 1.9150076499520017, 1.9318341666050154, 1.960158345012105, 1.942978864845533, 1.9662359381803116, 1.979614214766155, 1.9740790147225247, 1.9941553157679508, 2.032101814379844, 2.027958897267072, 2.103156582749824, 2.06489382259631, 2.1117389722836917, 2.012313402011025, 2.0615588573021246, 2.0966275015468483, 2.0410921387101713, 1.9958259105452685, 2.0172056604123725, 2.055103521821514, 2.1112841999169407, 2.1464159589078227, 2.1607556223677293, 2.215351240811989, 2.1665159975040487, 2.1057424081531044, 2.1077219063778365, 2.0793654806729234, 2.0905197964945046, 2.1442336508364286, 2.2164590568240943, 2.156841573939799, 2.0990060619564233, 2.0699580324512357, 2.005683324777918, 1.9253434404520964, 1.9389905433867805, 1.915965531740682, 1.9315091648470395, 1.9351150387132394, 1.9006459176826547, 1.8865369434659023, 1.909569769770867, 1.9113035410370576, 1.9316779055042972, 1.9313713395198495, 1.9505918484199603, 1.9177620526190942, 1.9366064092257926, 1.9049032775502144, 1.9246743042242702, 1.879756928751871, 1.8830937591100296, 1.890101066721217, 1.8871338841420542, 1.824959446835789, 1.761174699984402, 1.7663650453739854, 1.7375125564139717, 1.709789233117519, 1.6544755052557205, 1.6465222951209482, 1.6538922464254542, 1.6700223863006722, 1.6452828158870338, 1.6858999568777802, 1.6928441697544119, 1.7296401350826631, 1.775761503524008, 1.74730716164825, 1.762858617640147, 1.7465670995256615, 1.7297740475232266, 1.752509081076481]

        volume_product_constraint = [1.900781060821376, 1.9022833072875052, 1.9010674957067828, 1.9023200637568771, 1.901568370741048, 1.903466054890315, 1.9062439763787768, 1.9079206299199798, 1.9099486273900805, 1.911783286641779, 1.9125938586730926, 1.9149755681501117, 1.9165784320344357, 1.9167020087278255, 1.918633863443034, 1.9170999783390883, 1.9193920527332833, 1.920174971614161, 1.9219367354434382, 1.9227391954468116, 1.9209908070885207, 1.9202136281475062, 1.9193032460751414, 1.9197721816404827, 1.922047089013747, 1.9228252784832252, 1.9248818373516645, 1.9230229869804245, 1.9243649365762994, 1.924792042444172, 1.9263927638059397, 1.926711668830235, 1.925996002816589, 1.928195688861784, 1.9283842140401235, 1.9311815871023112, 1.9326508351251024, 1.931117958793459, 1.9282386136949763, 1.9253638419027188, 1.9271947313692088, 1.927544151683562, 1.9268375621271336, 1.9257279947338057, 1.9245175110297617, 1.9253810747740974, 1.9253039444164426, 1.9265260480232236, 1.9260999857462964, 1.9251206994595238, 1.9258013446265165, 1.9258335957174992, 1.9257870677316034, 1.926259050350641, 1.926921610901158, 1.927762579308305, 1.925467889925671, 1.9279667517784689, 1.9277863743364279, 1.9285264017180517, 1.9306895693205515, 1.9300955765625807, 1.9301936206498915, 1.9317496115340222, 1.9342311180803196, 1.9335320242860718, 1.9320183646032487, 1.9344848825523235, 1.934714652958108, 1.9327766249871094, 1.9318189164494173, 1.9298832908453976, 1.9317943980142085, 1.933156249634126, 1.9322803684606198, 1.9345588492232697, 1.9323989410489868, 1.9325158650819074, 1.9335401907287904, 1.935333086520489, 1.9373198367168611, 1.9395325982623217, 1.9396584756840813, 1.9377848079778586, 1.9399248542690135, 1.9399979291061968, 1.9402269840354933, 1.939051148177564]

        loading_constraint = [136.83418623597197, 134.7799828903343, 135.87752953724765, 137.13792284428806, 137.88242578029121, 137.7523489515728, 135.06172111067642, 134.85894726923289, 135.17856521548245, 132.28700101093924, 133.15091411689713, 132.8280520648643, 133.76598224218301, 133.6630099452608, 131.80086201771454, 131.8463851585635, 130.2292163956318, 130.57422991983447, 129.6667522703853, 129.14414109508292, 129.5858990037907, 128.6859534269726, 127.74998914459509, 127.45243527918178, 124.92768003631568, 125.64215017714889, 123.81944829169844, 126.59773740907988, 124.97449942587915, 123.71746331762091, 124.88461832375343, 126.13021321377485, 125.70772415067222, 124.10893328027622, 122.4948261028295, 120.85326220404438, 120.65647465065544, 120.07148266903741, 122.29247228554524, 124.9510338567927, 124.5448135114433, 125.41200113401725, 124.95474610315722, 123.96250151945355, 122.71096050945668, 124.01491969906893, 125.25993499365136, 125.63133914380045, 127.53648166808591, 129.7841921674624, 129.5412004805719, 130.2125541386931, 129.6948707981305, 129.77399607157923, 130.64272937977603, 130.96126446668322, 131.0795708071144, 129.93745566881088, 129.6861532375501, 129.04059302583914, 127.48622071038167, 128.4500132933783, 127.46466663081462, 128.12083159387316, 126.95989056509673, 128.7340054502955, 129.08818928381373, 128.19402900387743, 128.30313831437618, 131.00021871819877, 133.27858981277785, 133.48737359656226, 133.83239399872951, 134.34298466339456, 136.11298930547773, 135.7486679093971, 136.15998651537404, 135.46835136126478, 136.15049275606626, 134.76402170293304, 134.00865224651486, 132.38484056103806, 130.67110991841514, 132.0478883148335, 130.6470602202322, 131.5378756310326, 132.55638086268368, 132.19931866319737]


        max_mass = max(mass_constraint)
        max_loading = max(loading_constraint)

        mass_plot = [(i - AC.wing_mass_ub) / max_mass for i in mass_constraint]
        volume_plot = [volume_product_constraint[i] - volume_constraint[i] for i in range(len(volume_constraint))]
        loading_constraint = [(i - AC.wing_loading_ref * 1.1) / (max_loading) for i in loading_constraint]
        step = np.linspace(0, len(mass_constraint), len(mass_constraint))
        # print(len(step), len(clcd))

        # plt.plot(step, total_reward, color='#4292c6', label='Total rewards')
        # plt.plot(step, penalties, color='#ef6548', label='Penalties')
        # plt.plot(step, rewards, color="#78c679", label='Gains')
        # # plt.plot(step, volume_plot, color='grey', label='volume constraint')
        # plt.plot(step, CL_CD_list, color="#807dba", label='Objective value')
        # plt.hlines(0, 0, len(CL_CD_list), color="#737373", linestyles='dashed')
        # plt.xlabel('Objective value')


        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid('on', linestyle='dashed')
        plt.plot(step, mass_plot, color='#4292c6', label='Mass constraint')
        plt.plot(step, volume_plot, color='#ef6548', label='Volume constraint')
        plt.plot(step, loading_constraint, color="#78c679", label='Wing loading constraint')
        # plt.plot(step, CL_CD_list, color='orange', label='Objective')
        plt.hlines(0, 0, len(mass_constraint), color="#737373",linestyles='dashed')
        plt.vlines(63, -0.8, 0.4, colors="#636363", linestyles='dashed')
        fig.gca().legend(loc="upper left")
        plt.xlabel('Step')
        # plt.title(f'Rewards vs step')
        plt.show()
    #####################################################
    ####### TRANSFER LEARNING ###########
    #### ANALYSIS 10.5: Model Resilience
    # use analysis 2
    if ANALYSIS[9] == 1:
        resweight = Results('weight')
        PSO_CL_CD = [2.288800309938e+01, 2.514550165310e+01, 2.718948369009e+01, 2.916665091460e+01, 30.840,
                     3.226113275761e+01, 3.337784294547e+01, 3.429205308375e+01, 3.506395283760e+01]  # 9 PSO results
        obj_start = [21.75,23.96 , 25.73, 27.08, 28.08, 28.75, 29.17,29.36,29.37]
        fig = plt.figure(figsize=(10, 7))
        # ax1 = fig.add_subplot(111)

        # Creating axes instance
        # ax = fig.add_axes([0, 0, 1, 1])
        x_labels = ['-40%', '-30%', '-20%', '-10%', '0%', '+10%', '+20%', '+30%', '+40%']
        objectives_weight = []
        objectives_weight.append(resweight.minus40)
        objectives_weight.append(resweight.minus30)
        objectives_weight.append(resweight.minus20)
        objectives_weight.append(resweight.minus10)
        objectives_weight.append(res.objectives)
        objectives_weight.append(resweight.plus10)
        objectives_weight.append(resweight.plus20)
        objectives_weight.append(resweight.plus30)
        objectives_weight.append(resweight.plus40)

        PPO_AVERAGE = []
        PPO_MAX = []
        for res in range(len(objectives_weight)):
            # print(obej)
            ave = sum(objectives_weight[res]) / len(objectives_weight[res])
            max1 = max(objectives_weight[res])
            PPO_AVERAGE.append(ave)
            PPO_MAX.append(max1)


        # Creating plot
        # bp = ax1.boxplot(objectives_weight, labels=labels)
        x = np.linspace(1, 9, 9)
        # ax1.xticks(x, labels, rotation='vertical')
        ave_weight = []
        max_weight = []

        # for result in range(len(objectives_weight)):
        #     ave_weight.append(sum(objectives_weight[result]) / len(objectives_weight[result]))

            # plt.hlines(sum(objectives_weight[result])/len(objectives_weight[result]),
            #            0.75 + 1 * result, 1.25 + 1 * result, colors='blue')


        # plt.plot(step, total_reward, color='#4292c6', label='Total rewards')
        # plt.plot(step, penalties, color='#ef6548', label='Penalties')
        # plt.plot(step, rewards, color="#78c679", label='Gains')
        # # plt.plot(step, volume_plot, color='grey', label='volume constraint')
        # plt.plot(step, CL_CD_list, color="#807dba", label='Objective value')
        # plt.hlines(0, 0, len(CL_CD_list), color="#737373", linestyles='dashed')

        ax = plt.axes()
        # plt.plot(x, ave_weight, 'blue', label = 'PPO average objective')
        plt.plot(x, PSO_CL_CD, '#fd8d3c', label = 'PSO objective values')
        plt.plot(x, PPO_AVERAGE, "#78c679",label = 'PPO average objective values')
        plt.plot(x, PPO_MAX, '#4292c6', label = 'PPO maximum objective values')
        plt.plot(x, obj_start, "#737373", label = 'Starting objective values')

        plt.xlabel('Mass increments')
        plt.ylabel('Objective value')
        # plt.axis('off')
        ax.set_xticks(x, x_labels, rotation='horizontal')
        for var in range(len(PPO_AVERAGE)):
            plt.vlines(1 + 1 * var, obj_start[0], PPO_MAX[-1], colors="#969696", linestyles='dashed')

        # for pso in range(len(PSO_CL_CD)):
        #     plt.hlines(PSO_CL_CD[pso],
        #                0.75 + 1 * pso, 1.25 + 1 * pso, colors='green')
        # plt.hlines(0, 0.75, 9.25, 'black')
        # plt.hlines(1, 0.75, 9.25, 'black')
        # show plot
        plt.show()
        fig.gca().legend(loc="lower right")

    #### ANALYSIS 11: General performance of No retraining vs clean retraining vs continual learning vs transfer learning
    # Use analysis 2
    # Table x3 of the different base aircraft
    if ANALYSIS[10] == 1:
        def get_TL_PSO():
            aircraft_config = 'ASK21'
            directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
            results_PSO_list = os.listdir(directory_PSO)
            CL_CD_PSO = []
            design_vectors_PSO = []

            for f in range(len(results_PSO_list)):
                CL_CD = 0
                file_name = results_PSO_list[f]
                file_directory = directory_PSO + '/' + file_name

                with open(file_directory, 'r') as fp:
                    lines = len(fp.readlines())
                if lines > 0:
                    CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                    design_vector = []
                    for dv in range(9):
                        dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                        design_vector.append(-dv_i)
                design_vectors_PSO.append(design_vector)
                CL_CD_PSO.append(CL_CD)

            sorted_CL_CD_PSO = sorted(CL_CD_PSO)
            best_CL_CD_PSO = sorted_CL_CD_PSO[-10:]
            best_CL_CD_PSO.reverse()
            best_idx_PSO = []
            best_design_PSO = []

            for idx in range(len(best_CL_CD_PSO)):
                best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

            for idx in range(len(best_CL_CD_PSO)):
                best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])
            return best_CL_CD_PSO, best_design_PSO


        CL_CD, DV = get_TL_PSO()
        print('best', max(CL_CD))
        print('ave', sum(CL_CD) / len(CL_CD))
        print('std', np.std(CL_CD))

    #### ANALYSIS 12: planforms TL DA50
    # Use analysis 3
    if ANALYSIS[11] == 1:
        # SR22 optimized
        # PSO optimized
        # Base max
        # TL
        # CL
        design_vector_start = [AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start,
                               AC.sweep_le_start, AC.yoffset_kink_start, AC.zoffset_tip_start,
                               AC.twist_start, AC.thickness_chord_start]

        # SORTING ML RESULTS

        # PLOTTING planforms results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        plt.axis('equal')
        # plt.title(f'Optimized DA50 wing planforms')
        x_AC1, y_AC1 = points(design_vector_start, AC.fuselage_width)
        label1 = 'Starting planform'
        plt.plot(x_AC1[:8], y_AC1[:8], color='#969696', label=label1)
        plt.vlines(0, -2, 2.8, colors='#f7fbff',linestyles='dashed')
        plt.vlines(0, -2, 0.8, colors='#525252',linestyles='dashed')





        DA50_BASE = [15., 1.12625848, 1.12412871, 1.12412871, 0., 2.38313326,
                     0.70087903, -1.7692592, 0.14296191]
        x_AC2, y_AC2 = points(DA50_BASE, AC.fuselage_width)
        label2 = 'DA50 model'
        plt.plot(x_AC2[:8], y_AC2[:8], label=label2, color= '#4292c6')

        DA50_TL = [15., 1.11163634, 1.11163634, 1.11163634, 0., 2.2576664,
                   0., -2.24588247, 0.14620796]
        x_AC3, y_AC3 = points(DA50_TL, AC.fuselage_width)
        label3 = 'TL model'
        plt.plot(x_AC3[:8], y_AC3[:8], label=label3, color= "#807dba")

        DA50_CL = [15., 1.10179757, 1.10179757, 1.09788274, 0., 2.50084696,
                   0., -1.89337647, 0.14910475]
        x_AC4, y_AC4 = points(DA50_CL, AC.fuselage_width)
        label4 = 'CL model'
        plt.plot(x_AC4[:8], y_AC4[:8], label=label4, color= "#78c679")

        DA50_SR22 = [15., 1.14127798, 1.14127798, 1.14127798, 0.20849882, 2.43142124,
                     0., -1.19255368, 0.13904111]
        x_AC5, y_AC5 = points(DA50_SR22, AC.fuselage_width)
        label5 = 'SR22 model'
        plt.plot(x_AC5[:8], y_AC5[:8], label=label5, color= "#737373")

        #####PSO PLOTTING
        number_PSO = 1
        aircraft_config = 'DA50'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        sorted_CL_CD_PSO = sorted(CL_CD_PSO)
        best_CL_CD_PSO = sorted_CL_CD_PSO[-10:]
        best_CL_CD_PSO.reverse()
        best_idx_PSO = []
        best_design_PSO = []

        for idx in range(len(best_CL_CD_PSO)):
            best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

        for idx in range(len(best_CL_CD_PSO)):
            best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

        # PLOTTING ML results
        x_AC_PSO = []
        y_AC_PSO = []

        for design in range(number_PSO):
            x_ACi, y_ACi = points(best_design_PSO[design], AC.fuselage_width)
            x_AC_PSO.append(x_ACi)
            y_AC_PSO.append(y_ACi)
            # labeli = f'PSO design {design + 1}, CL/CD:{round(best_CL_CD_ML[design],3)}'
            plt.plot(x_AC_PSO[design][:8], y_AC_PSO[design][:8], color='#ef6548', label='PSO')

        fig.gca().legend(loc="upper right")
        # fig.axes.yaxis.set_ticklabels([])
        plt.show()

    #### ANALYSIS 13: design variables difference
    # Look at the delta DVi of the original model compared with the transfered model to estimate adaptability
    # use analysis 4 >
    if ANALYSIS[12] == 1:
        objectives = res.objectives
        design_vectors = res.design_vectors
        collected_design_variables = []
        normalized_design_variables = []
        design_vector_start = (
        AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start, AC.sweep_le_start,
        AC.yoffset_kink_start, AC.zoffset_tip_start, AC.twist_start, AC.thickness_chord_start)

        #####Get PSO
        number_PSO = 1
        aircraft_config = 'SR22'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        sorted_CL_CD_PSO = sorted(CL_CD_PSO)
        best_CL_CD_PSO = sorted_CL_CD_PSO[-number_PSO:]
        best_CL_CD_PSO.reverse()
        best_idx_PSO = []
        best_design_PSO = []

        for idx in range(len(best_CL_CD_PSO)):
            best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

        for idx in range(len(best_CL_CD_PSO)):
            best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

        dv_DA50_SR22 = [15., 1.14127798, 1.14127798, 1.14127798, 0.20849882, 2.43142124,
                        0., -1.19255368, 0.13904111]

        dv_DA50_base = [15., 1.12625848, 1.12412871, 1.12412871, 0., 2.38313326,
                        0.70087903, -1.7692592, 0.14296191]
        dv_DA50_TL = [15., 1.11163634, 1.11163634, 1.11163634, 0., 2.2576664,
                      0., -2.24588247, 0.14620796]
        dv_DA50_CL = [15., 1.10179757, 1.10179757, 1.09788274, 0., 2.50084696, 0., -1.89337647, 0.14910475]


        def get_div(design_vector):
            div_vector = []
            for dv in range(len(design_vector)):
                div = design_vector[dv] - design_vector_start[dv]

                div_vector.append(div)
            return div_vector


        div_SR22 = get_div(dv_DA50_SR22)
        div_base = get_div(dv_DA50_base)
        div_TL = get_div(dv_DA50_TL)
        div_CL = get_div(dv_DA50_CL)
        div_PSO = get_div(best_design_PSO[0])

        fig = plt.figure(figsize=(10, 7))
        # ax1 = fig.add_subplot(111)
        ax = plt.axes()

        # Creating axes instance
        # ax = fig.add_axes([0, 0, 1, 1])
        labels = [r'$b$', r'$c_{root}$', r'$c_{kink}$', r'$c_{tip}$', r'$\Lambda_{LE}$', r'$y_{kink}$', r'$z_{tip}$', r'$\epsilon$', r'$t/c$']
        # Creating plot
        # bp = ax1.boxplot(normalized_design_variables, labels=labels)
        x = np.linspace(1, 9, 9)
        plt.plot(x, div_SR22, label='SR22 base model', color="#737373")
        plt.plot(x, div_base, label='New model', color='#4292c6')
        plt.plot(x, div_TL, label='TL model', color="#807dba")
        plt.plot(x, div_CL, label='CL model', color="#78c679")
        plt.plot(x, div_PSO, label='PSO', color='#ef6548')
        ax.set_xticks(x, labels, rotation='horizontal')

        # Starting lines

        for var in range(len(design_vector_start)):
            plt.vlines(1 + 1 * var, -2, 2, colors="#737373", linestyles='dashed')
        plt.hlines(0, 0.75, 9.25, "#737373")
        # plt.vlines(45.5, -0.1, 1.1, colors="#636363", linestyles='dashed')
        # plt.hlines(1, 0.75, 9.25, 'black')
        # # show plot
        fig.gca().legend(loc="upper right")
        plt.show()

    if ANALYSIS[12] == 2:
        if ANALYSIS[12] == 1:
            objectives = res.objectives
            design_vectors = res.design_vectors
            collected_design_variables = []
            normalized_design_variables = []
            design_vector_start = (
                AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start, AC.sweep_le_start,
                AC.yoffset_kink_start, AC.zoffset_tip_start, AC.twist_start, AC.thickness_chord_start)

            #####Get PSO
            number_PSO = 1
            aircraft_config = 'DA50'
            directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
            results_PSO_list = os.listdir(directory_PSO)
            CL_CD_PSO = []
            design_vectors_PSO = []



            for f in range(len(results_PSO_list)):
                CL_CD = 0
                file_name = results_PSO_list[f]
                file_directory = directory_PSO + '/' + file_name

                with open(file_directory, 'r') as fp:
                    lines = len(fp.readlines())
                if lines > 0:
                    CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                    design_vector = []
                    for dv in range(9):
                        dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                        design_vector.append(-dv_i)
                design_vectors_PSO.append(design_vector)
                CL_CD_PSO.append(CL_CD)

            sorted_CL_CD_PSO = sorted(CL_CD_PSO)
            best_CL_CD_PSO = sorted_CL_CD_PSO[-number_PSO:]
            best_CL_CD_PSO.reverse()
            best_idx_PSO = []
            best_design_PSO = []

            for idx in range(len(best_CL_CD_PSO)):
                best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

            for idx in range(len(best_CL_CD_PSO)):
                best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

            dv_DA50_SR22 = [15., 1.14127798, 1.14127798, 1.14127798, 0.20849882, 2.43142124,
                            0., -1.19255368, 0.13904111]

            dv_DA50_base = [15., 1.12625848, 1.12412871, 1.12412871, 0., 2.38313326,
                            0.70087903, -1.7692592, 0.14296191]
            dv_DA50_TL = [15., 1.11163634, 1.11163634, 1.11163634, 0., 2.2576664,
                          0., -2.24588247, 0.14620796]
            dv_DA50_CL = [15., 1.10179757, 1.10179757, 1.09788274, 0., 2.50084696, 0., -1.89337647, 0.14910475]


            def get_div(design_vector):
                div_vector = []
                for dv in range(len(design_vector)):
                    div = design_vector[dv] - design_vector_start[dv]

                    div_vector.append(div)
                return div_vector


            div_SR22 = get_div(dv_DA50_SR22)
            div_base = get_div(dv_DA50_base)
            div_TL = get_div(dv_DA50_TL)
            div_CL = get_div(dv_DA50_CL)
            div_PSO = get_div(best_design_PSO[0])

            fig = plt.figure(figsize=(10, 7))
            # ax1 = fig.add_subplot(111)
            ax = plt.axes()

            # Creating axes instance
            # ax = fig.add_axes([0, 0, 1, 1])
            labels = [r'$b$', r'$c_{root}$', r'$c_{kink}$', r'$c_{tip}$', r'$\Lambda_{LE}$', r'$y_{kink}$',
                      r'$z_{tip}$', r'$\epsilon$', r'$t/c$']
            # Creating plot
            # bp = ax1.boxplot(normalized_design_variables, labels=labels)
            x = np.linspace(1, 9, 9)
            plt.plot(x, div_SR22, label='SR22 base model', color="#737373")
            plt.plot(x, div_base, label='New model', color='#4292c6')
            plt.plot(x, div_TL, label='TL model', color="#807dba")
            plt.plot(x, div_CL, label='CL model', color="#78c679")
            plt.plot(x, div_PSO, label='PSO', color='#ef6548')
            ax.set_xticks(x, labels, rotation='horizontal')

            # Starting lines

            for var in range(len(design_vector_start)):
                plt.vlines(1 + 1 * var, -2, 2, colors="#737373", linestyles='dashed')
            plt.hlines(0, 0.75, 9.25, "#737373")
            plt.vlines
            # plt.hlines(1, 0.75, 9.25, 'black')
            # # show plot
            fig.gca().legend(loc="upper right")
            plt.show()

    if ANALYSIS[12] == 3:
        #Average design vector changes
        objectives = res.objectives
        design_vectors = res.design_vectors
        collected_design_variables = []
        normalized_design_variables = []
        design_vector_start = (
        AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start, AC.sweep_le_start,
        AC.yoffset_kink_start, AC.zoffset_tip_start, AC.twist_start, AC.thickness_chord_start)

        #####Get PSO
        number_PSO = 10
        aircraft_config = 'DA50'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        average_PSO_DV = []

        def get_ave(design_vector_collection):
            average_DV = []
            for var in range(len(design_vector_collection[0])):
                var_list = []
                for dv in range(len(design_vector_collection)):
                    var_list.append(design_vector_collection[dv][var])

                ave_var = sum(var_list)/len(var_list)
                average_DV.append(ave_var)
            return average_DV

        average_PSO_DV = get_ave(design_vectors_PSO)

        sorted_CL_CD_PSO = sorted(CL_CD_PSO)
        best_CL_CD_PSO = sorted_CL_CD_PSO[-number_PSO:]
        best_CL_CD_PSO.reverse()
        best_idx_PSO = []
        best_design_PSO = []

        for idx in range(len(best_CL_CD_PSO)):
            best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

        for idx in range(len(best_CL_CD_PSO)):
            best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

        #Get the dvs from the results file and average them
        res_da50 = Results('DA_50')
        dv_DA50_SR22 = get_ave(res_da50.dv_DA50_SR22)

        dv_DA50_base = get_ave(res_da50.dv_DA50_base)
        dv_DA50_TL = get_ave(res_da50.dv_DA50_TL)
        dv_DA50_CL = get_ave(res_da50.dv_DA50_CL)


        def get_div(design_vector):
            div_vector = []
            for dv in range(len(design_vector)):
                div = design_vector[dv] - design_vector_start[dv]

                div_vector.append(div)
            return div_vector


        div_SR22 = get_div(dv_DA50_SR22)
        div_base = get_div(dv_DA50_base)
        div_TL = get_div(dv_DA50_TL)
        div_CL = get_div(dv_DA50_CL)
        div_PSO = get_div(best_design_PSO[0])

        fig = plt.figure(figsize=(10, 7))
        # ax1 = fig.add_subplot(111)
        ax = plt.axes()

        # Creating axes instance
        # ax = fig.add_axes([0, 0, 1, 1])
        labels = [r'$b$', r'$c_{root}$', r'$c_{kink}$', r'$c_{tip}$', r'$\Lambda_{LE}$', r'$y_{kink}$', r'$z_{tip}$', r'$\epsilon$', r'$t/c$']
        # Creating plot
        # bp = ax1.boxplot(normalized_design_variables, labels=labels)
        x = np.linspace(1, 9, 9)
        plt.plot(x, div_SR22, label='SR22 base model', color="#737373")
        plt.plot(x, div_base, label='DA50 model', color='#4292c6')
        plt.plot(x, div_TL, label='TL model', color="#807dba")
        plt.plot(x, div_CL, label='CL model', color="#78c679")
        plt.plot(x, div_PSO, label='PSO', color='#ef6548')
        ax.set_xticks(x, labels, rotation='horizontal')

        # Starting lines

        for var in range(len(design_vector_start)):
            plt.vlines(1 + 1 * var, -2, 2, colors="#737373", linestyles='dashed')
        plt.hlines(0, 0.75, 9.25, "#737373")
        plt.vlines
        # plt.hlines(1, 0.75, 9.25, 'black')
        # # show plot
        fig.gca().legend(loc="upper right")
        plt.show()

    if ANALYSIS[12] == 4:
        #Average design vector changes ASK21
        objectives = res.objectives
        design_vectors = res.design_vectors
        collected_design_variables = []
        normalized_design_variables = []
        design_vector_start = (
        AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start, AC.sweep_le_start,
        AC.yoffset_kink_start, AC.zoffset_tip_start, AC.twist_start, AC.thickness_chord_start)

        #####Get PSO
        number_PSO = 10
        aircraft_config = 'F50'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        average_PSO_DV = []

        def get_ave(design_vector_collection):
            average_DV = []
            for var in range(len(design_vector_collection[0])):
                var_list = []
                for dv in range(len(design_vector_collection)):
                    var_list.append(design_vector_collection[dv][var])

                ave_var = sum(var_list)/len(var_list)
                average_DV.append(ave_var)
            return average_DV

        average_PSO_DV = get_ave(design_vectors_PSO)

        sorted_CL_CD_PSO = sorted(CL_CD_PSO)
        best_CL_CD_PSO = sorted_CL_CD_PSO[-number_PSO:]
        best_CL_CD_PSO.reverse()
        best_idx_PSO = []
        best_design_PSO = []

        for idx in range(len(best_CL_CD_PSO)):
            best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

        for idx in range(len(best_CL_CD_PSO)):
            best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

        #Get the dvs from the results file and average them
        res_da50 = Results('F_50')
        dv_DA50_SR22 = get_ave(res_da50.dv_F50_SR22)
        dv_DA50_base = get_ave(res_da50.dv_F50_base)
        dv_DA50_TL = get_ave(res_da50.dv_F50_TL)
        dv_DA50_CL = get_ave(res_da50.dv_F50_CL)


        def get_div(design_vector):
            div_vector = []
            for dv in range(len(design_vector)):
                div = design_vector[dv] - design_vector_start[dv]

                div_vector.append(div)
            return div_vector


        div_SR22 = get_div(dv_DA50_SR22)
        div_base = get_div(dv_DA50_base)
        div_TL = get_div(dv_DA50_TL)
        div_CL = get_div(dv_DA50_CL)
        div_PSO = get_div(best_design_PSO[0])

        fig = plt.figure(figsize=(10, 7))
        # ax1 = fig.add_subplot(111)
        ax = plt.axes()

        # Creating axes instance
        # ax = fig.add_axes([0, 0, 1, 1])
        labels = [r'$b$', r'$c_{root}$', r'$c_{kink}$', r'$c_{tip}$', r'$\Lambda_{LE}$', r'$y_{kink}$', r'$z_{tip}$', r'$\epsilon$', r'$t/c$']
        # Creating plot
        # bp = ax1.boxplot(normalized_design_variables, labels=labels)
        x = np.linspace(1, 9, 9)
        plt.plot(x, div_SR22, label='SR22 base model', color="#737373")
        plt.plot(x, div_base, label='New model', color='#4292c6')
        plt.plot(x, div_TL, label='TL model', color="#807dba")
        plt.plot(x, div_CL, label='CL model', color="#78c679")
        plt.plot(x, div_PSO, label='PSO', color='#ef6548')
        ax.set_xticks(x, labels, rotation='horizontal')

        # Starting lines

        for var in range(len(design_vector_start)):
            plt.vlines(1 + 1 * var, -4, 7, colors="#737373", linestyles='dashed')
        plt.hlines(0, 0.75, 9.25, "#737373")
        plt.vlines
        # plt.hlines(1, 0.75, 9.25, 'black')
        # # show plot
        fig.gca().legend(loc="upper right")
        plt.show()

    if ANALYSIS[12] == 5:
        #Average design vector changes
        objectives = res.objectives
        design_vectors = res.design_vectors
        collected_design_variables = []
        normalized_design_variables = []
        design_vector_start = (
        AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start, AC.sweep_le_start,
        AC.yoffset_kink_start, AC.zoffset_tip_start, AC.twist_start, AC.thickness_chord_start)

        #####Get PSO
        number_PSO = 10
        aircraft_config = 'DA50'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        average_PSO_DV = []

        def get_ave(design_vector_collection):
            average_DV = []
            for var in range(len(design_vector_collection[0])):
                var_list = []
                for dv in range(len(design_vector_collection)):
                    var_list.append(design_vector_collection[dv][var])

                ave_var = sum(var_list)/len(var_list)
                average_DV.append(ave_var)
            return average_DV

        average_PSO_DV = get_ave(design_vectors_PSO)

        sorted_CL_CD_PSO = sorted(CL_CD_PSO)
        best_CL_CD_PSO = sorted_CL_CD_PSO[-number_PSO:]
        best_CL_CD_PSO.reverse()
        best_idx_PSO = []
        best_design_PSO = []

        for idx in range(len(best_CL_CD_PSO)):
            best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

        for idx in range(len(best_CL_CD_PSO)):
            best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

        #Get the dvs from the results file and average them
        res_da50 = Results('DA_50')
        dv_DA50_SR22 = get_ave(res_da50.dv_DA50_SR22)

        dv_DA50_base = get_ave(res_da50.dv_DA50_base)
        dv_DA50_TL = get_ave(res_da50.dv_DA50_TL)
        dv_DA50_CL = get_ave(res_da50.dv_DA50_CL)


        def get_div(design_vector):
            div_vector = []
            for dv in range(len(design_vector)):
                div = design_vector[dv] - design_vector_start[dv]

                div_vector.append(div)
            return div_vector


        div_SR22 = get_div(dv_DA50_SR22)
        div_base = get_div(dv_DA50_base)
        div_TL = get_div(dv_DA50_TL)
        div_CL = get_div(dv_DA50_CL)
        div_PSO = get_div(best_design_PSO[0])

        fig = plt.figure(figsize=(10, 7))
        # ax1 = fig.add_subplot(111)
        ax = plt.axes()

        # Creating axes instance
        # ax = fig.add_axes([0, 0, 1, 1])
        labels = [r'$b$', r'$c_{root}$', r'$c_{kink}$', r'$c_{tip}$', r'$\Lambda_{LE}$', r'$y_{kink}$', r'$z_{tip}$', r'$\epsilon$', r'$t/c$']
        # Creating plot
        # bp = ax1.boxplot(normalized_design_variables, labels=labels)
        x = np.linspace(1, 9, 9)
        plt.plot(x, div_SR22, label='SR22 base model', color="#737373")
        plt.plot(x, div_base, label='New model', color='#4292c6')
        plt.plot(x, div_TL, label='TL model', color="#807dba")
        plt.plot(x, div_CL, label='CL model', color="#78c679")
        plt.plot(x, div_PSO, label='PSO', color='#ef6548')
        ax.set_xticks(x, labels, rotation='horizontal')

        # Starting lines

        for var in range(len(design_vector_start)):
            plt.vlines(1 + 1 * var, -2, 2, colors="#737373", linestyles='dashed')
        plt.hlines(0, 0.75, 9.25, "#737373")
        plt.vlines
        # plt.hlines(1, 0.75, 9.25, 'black')
        # # show plot
        fig.gca().legend(loc="upper right")
        plt.show()


    if ANALYSIS[12] == 6:
        #DA50 new plot
        #Average design vector changes
        objectives = res.objectives
        design_vectors = res.design_vectors
        collected_design_variables = []
        normalized_design_variables = []
        design_vector_start = (
        AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start, AC.sweep_le_start,
        AC.yoffset_kink_start, AC.zoffset_tip_start, AC.twist_start, AC.thickness_chord_start)

        #####Get PSO
        number_PSO = 10
        aircraft_config = 'DA50'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        average_PSO_DV = []

        def get_ave(design_vector_collection):
            average_DV = []
            for var in range(len(design_vector_collection[0])):
                var_list = []
                for dv in range(len(design_vector_collection)):
                    var_list.append(design_vector_collection[dv][var])

                ave_var = sum(var_list)/len(var_list)
                average_DV.append(ave_var)
            return average_DV

        average_PSO_DV = get_ave(design_vectors_PSO)

        sorted_CL_CD_PSO = sorted(CL_CD_PSO)
        best_CL_CD_PSO = sorted_CL_CD_PSO[-number_PSO:]
        best_CL_CD_PSO.reverse()
        best_idx_PSO = []
        best_design_PSO = []

        for idx in range(len(best_CL_CD_PSO)):
            best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

        for idx in range(len(best_CL_CD_PSO)):
            best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

        #Get the dvs from the results file and average them
        res_da50 = Results('DA_50')
        dv_DA50_SR22 = get_ave(res_da50.dv_DA50_SR22)

        dv_DA50_base = get_ave(res_da50.dv_DA50_base)
        dv_DA50_TL = get_ave(res_da50.dv_DA50_TL)
        dv_DA50_CL = get_ave(res_da50.dv_DA50_CL)


        def get_div(design_vector):
            div_vector = []
            for dv in range(len(design_vector)):
                div = design_vector[dv] - design_vector_start[dv]

                div_vector.append(div)
            return div_vector


        div_SR22 = get_div(dv_DA50_SR22)
        div_base = get_div(dv_DA50_base)
        div_TL = get_div(dv_DA50_TL)
        div_CL = get_div(dv_DA50_CL)
        div_PSO = get_div(best_design_PSO[0])

        fig = plt.figure(figsize=(10, 7))
        # ax1 = fig.add_subplot(111)
        ax = plt.axes()

        # Creating axes instance
        # ax = fig.add_axes([0, 0, 1, 1])
        labels = [r'$b$', r'$c_{root}$', r'$c_{kink}$', r'$c_{tip}$', r'$\Lambda_{LE}$', r'$y_{kink}$', r'$z_{tip}$', r'$\epsilon$', r'$t/c$']
        # Creating plot
        # bp = ax1.boxplot(normalized_design_variables, labels=labels)
        x = np.linspace(1, 9, 9)

        for var in range(len(div_SR22)):
            plt.hlines(div_SR22[var],0.5 +1 * var , 1.5 + 1 * var  , color="#252525")
        for var in range(len(div_SR22)):
            plt.hlines(div_base[var],0.5 +1 * var , 1.5 + 1 * var  , color='#4292c6')
        for var in range(len(div_SR22)):
            plt.hlines(div_TL[var],0.5 +1 * var , 1.5 + 1 * var  , color="#807dba")
        for var in range(len(div_SR22)):
            plt.hlines(div_CL[var],0.5 +1 * var , 1.5 + 1 * var  , color="#78c679")
        for var in range(len(div_SR22)):
            plt.hlines(div_PSO[var],0.5 +1 * var , 1.5 + 1 * var  , color='#ef6548')



        plt.plot(1, 0, label='SR22 base model', color="#737373")
        plt.plot(1, 0, label='DA50 model', color='#4292c6')
        plt.plot(1, 0, label='TL model', color="#807dba")
        plt.plot(1,0, label='CL model', color="#78c679")
        plt.plot(1, 0, label='PSO', color='#ef6548')
        ax.set_xticks(x, labels, rotation='horizontal')

        # Starting lines

        for var in range(len(design_vector_start)+1):
            plt.vlines(0.5 + 1 * var, -2.4, 2.4, colors="#737373", linestyles='dashed')

        plt.hlines(0, 0.5, 9.25, color ="#737373")

        # plt.hlines(1, 0.75, 9.25, 'black')
        # # show plot
        fig.gca().legend(loc="upper right")
        plt.show()


    if ANALYSIS[12] == 7:
        #Average design vector changes F50
        objectives = res.objectives
        design_vectors = res.design_vectors
        collected_design_variables = []
        normalized_design_variables = []
        design_vector_start = (
        AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start, AC.sweep_le_start,
        AC.yoffset_kink_start, AC.zoffset_tip_start, AC.twist_start, AC.thickness_chord_start)

        #####Get PSO
        number_PSO = 10
        aircraft_config = 'ASK21'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        average_PSO_DV = []

        def get_ave(design_vector_collection):
            average_DV = []
            for var in range(len(design_vector_collection[0])):
                var_list = []
                for dv in range(len(design_vector_collection)):
                    var_list.append(design_vector_collection[dv][var])

                ave_var = sum(var_list)/len(var_list)
                average_DV.append(ave_var)
            return average_DV

        average_PSO_DV = get_ave(design_vectors_PSO)

        sorted_CL_CD_PSO = sorted(CL_CD_PSO)
        best_CL_CD_PSO = sorted_CL_CD_PSO[-number_PSO:]
        best_CL_CD_PSO.reverse()
        best_idx_PSO = []
        best_design_PSO = []

        for idx in range(len(best_CL_CD_PSO)):
            best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

        for idx in range(len(best_CL_CD_PSO)):
            best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

        #Get the dvs from the results file and average them
        res_da50 = Results('ASK_21')
        dv_DA50_SR22 = get_ave(res_da50.dv_ASK21_SR22)
        dv_DA50_base = get_ave(res_da50.dv_ASK21_base)
        dv_DA50_TL = get_ave(res_da50.dv_ASK21_TL)
        dv_DA50_CL = get_ave(res_da50.dv_ASK21_CL)


        def get_div(design_vector):
            div_vector = []
            for dv in range(len(design_vector)):
                div = design_vector[dv] - design_vector_start[dv]

                div_vector.append(div)
            return div_vector


        div_SR22 = get_div(dv_DA50_SR22)
        div_base = get_div(dv_DA50_base)
        div_TL = get_div(dv_DA50_TL)
        div_CL = get_div(dv_DA50_CL)
        div_PSO = get_div(best_design_PSO[0])

        fig = plt.figure(figsize=(10, 7))
        # ax1 = fig.add_subplot(111)
        ax = plt.axes()

        # Creating axes instance
        # ax = fig.add_axes([0, 0, 1, 1])
        labels = [r'$b$', r'$c_{root}$', r'$c_{kink}$', r'$c_{tip}$', r'$\Lambda_{LE}$', r'$y_{kink}$', r'$z_{tip}$', r'$\epsilon$', r'$t/c$']
        # Creating plot
        # bp = ax1.boxplot(normalized_design_variables, labels=labels)
        x = np.linspace(1, 9, 9)


        for var in range(len(div_SR22)):
            plt.hlines(div_SR22[var],0.5 +1 * var , 1.5 + 1 * var  , color="#252525")
        for var in range(len(div_SR22)):
            plt.hlines(div_base[var],0.5 +1 * var , 1.5 + 1 * var  , color='#4292c6')
        for var in range(len(div_SR22)):
            plt.hlines(div_TL[var],0.5 +1 * var , 1.5 + 1 * var  , color="#807dba")
        for var in range(len(div_SR22)):
            plt.hlines(div_CL[var],0.5 +1 * var , 1.5 + 1 * var  , color="#78c679")
        for var in range(len(div_SR22)):
            plt.hlines(div_PSO[var],0.5 +1 * var , 1.5 + 1 * var  , color='#ef6548')


        ax.set_xticks(x, labels, rotation='horizontal')
        plt.plot(1, 0, label='SR22 base model', color="#737373")
        plt.plot(1, 0, label='ASK21 model', color='#4292c6')
        plt.plot(1, 0, label='TL model', color="#807dba")
        plt.plot(1,0, label='CL model', color="#78c679")
        plt.plot(1, 0, label='PSO', color='#ef6548')
        ax.set_xticks(x, labels, rotation='horizontal')

        # Starting lines

        for var in range(len(design_vector_start) + 1):
            plt.vlines(0.5 + 1 * var, -4, 6, colors="#737373", linestyles='dashed')

        plt.hlines(0, 0.5, 9.5, color="#737373")
        # # show plot
        fig.gca().legend(loc="upper right")
        plt.show()




    #### ANALYSIS 14: rewards vs objective
    # use analysis 8 for best model

    if ANALYSIS[13] == 1:
        x = 2

    #### ANALYSIS 15: TL DA50 planforms
    # Get results from at 10% and 20% weight increase and decrease and compare with PSO

    if ANALYSIS[14] == 1:
        # SR22 optimized
        # PSO optimized
        # Base max
        # TL
        # CL
        design_vector_start = [AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start,
                               AC.sweep_le_start, AC.yoffset_kink_start, AC.zoffset_tip_start,
                               AC.twist_start, AC.thickness_chord_start]

        # SORTING ML RESULTS

        # PLOTTING planforms results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        plt.axis('equal')
        # plt.title(f'Optimized DA50 wing planforms at increase')

        x_AC1, y_AC1 = points(design_vector_start, AC.fuselage_width)
        label1 = 'Starting planform'
        plt.plot(x_AC1[:8], y_AC1[:8], color='#737373', label=label1)

        DA50_SR22 = [15., 1.14127798, 1.14127798, 1.14127798, 0.20849882, 2.43142124,
                     0., -1.19255368, 0.13904111]
        x_AC3, y_AC3 = points(DA50_SR22, AC.fuselage_width)
        label3 = 'SR22 model'
        plt.plot(x_AC3[:8], y_AC3[:8], label=label3, color="#807dba")

        DA50_13 = [13.71028515, 1.68377078, 1.33513457, 0.93114424, 1.52823878, 1.91432796,
                   0.53943181, 0.34916527, 0.13216044]
        x_AC4, y_AC4 = points(DA50_13, AC.fuselage_width)
        label4 = '1 300 000 steps'
        plt.plot(x_AC4[:8], y_AC4[:8], label=label4, color="#8c510a")

        DA50_14 = [15., 1.27292361, 1.27292361, 1.27292361, 0., 2.04395896,
                   0.10326508, -1.98513968, 0.11296032]
        x_AC5, y_AC5 = points(DA50_14, AC.fuselage_width)
        label5 = '1 400 000 steps'
        plt.plot(x_AC5[:8], y_AC5[:8], label=label5, color='#f46d43')

        DA50_15 = [1.50000000e+01, 1.10483824e+00, 1.10483824e+00, 1.10483824e+00,
                   1.33358081e-02, 2.29095640e+00, 0.00000000e+00, -2.37693035e+00,
                   1.47982073e-01]
        x_AC6, y_AC6 = points(DA50_15, AC.fuselage_width)
        label6 = '1 500 000 steps'
        plt.plot(x_AC6[:8], y_AC6[:8], label=label6, color='#fee08b')

        DA50_16 = [15., 1.13897098, 1.13897098, 1.1345177, 0., 2.23179305,
                   0., -1.93698805, 0.13990738]
        x_AC7, y_AC7 = points(DA50_16, AC.fuselage_width)
        label7 = '1 600 000 steps'
        plt.plot(x_AC7[:8], y_AC7[:8], label=label7, color='#abdda4')

        DA50_17 = [15., 1.10082334, 1.10082334, 1.10082334, 0., 2.26113497,
                   0., -2.07258035, 0.14897146]
        x_AC8, y_AC8 = points(DA50_17, AC.fuselage_width)
        label8 = '1 700 000 steps'
        plt.plot(x_AC8[:8], y_AC8[:8], label=label8, color='#3288bd')

        DA50_last = [15., 1.12625848, 1.12412871, 1.12412871, 0., 2.38313326,
                     0.70087903, -1.7692592, 0.14296191]
        x_AC2, y_AC2 = points(DA50_last, AC.fuselage_width)
        label2 = '1 800 000 steps'
        plt.plot(x_AC2[:8], y_AC2[:8], label=label2, color='#01665e')

        plt.vlines(0, -2, 2.8, colors='#f7fbff',linestyles='dashed')
        plt.vlines(0, -2, 1, colors='#525252', linestyles='dashed')
        fig.gca().legend(loc="upper right")
        # fig.axes.yaxis.set_ticklabels([])
        plt.show()
    if ANALYSIS[14] == 2:
        # SR22 optimized
        # PSO optimized
        # Base max
        # TL
        # CL
        design_vector_start = [AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start,
                               AC.sweep_le_start, AC.yoffset_kink_start, AC.zoffset_tip_start,
                               AC.twist_start, AC.thickness_chord_start]

        # SORTING ML RESULTS

        # PLOTTING planforms results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        plt.axis('equal')
        # plt.title(f'Optimized DA50 wing planforms at increase')

        x_AC1, y_AC1 = points(design_vector_start, AC.fuselage_width)
        label1 = 'Starting planform'
        plt.plot(x_AC1[:8], y_AC1[:8], color='#737373', label=label1)

        DA50_SR22 = [15.   ,       1.46427051 , 1.46427051 , 0.43048479 , 0.     ,     3.65997055,
  0.52094273 ,-0.41330322,  0.14177881]
        x_AC3, y_AC3 = points(DA50_SR22, AC.fuselage_width)
        label3 = '10 000 steps'
        plt.plot(x_AC3[:8], y_AC3[:8], label=label3, color='#8c510a')

  #       DA50_13 = [15.    ,      1.54530973,  1.48507243, 0.41597277,  0.     ,     2.84813003,
  # 0.26072752,  1.06095049  ,0.14924711]
  #       x_AC4, y_AC4 = points(DA50_13, AC.fuselage_width)
  #       label4 = '26 000 steps'
  #       plt.plot(x_AC4[:8], y_AC4[:8], label=label4, color="#045a8d")

        DA50_14 = [15.    ,      1.4119009 ,  1.4119009 ,  0.81284047,  0.   ,       2.60248948,
  0.49132587 ,-0.09175408  ,0.14401339]
        x_AC5, y_AC5 = points(DA50_14, AC.fuselage_width)
        label5 = '50 000 steps'
        plt.plot(x_AC5[:8], y_AC5[:8], label=label5, color='#f46d43')

        DA50_15 = [15.       ,   1.34949942 , 1.34949942  ,0.82683947 , 1.56299034 , 3.46338296,
  0.43042348, -0.74903987 , 0.1452902 ]
        x_AC6, y_AC6 = points(DA50_15, AC.fuselage_width)
        label6 = '100 000 steps'
        plt.plot(x_AC6[:8], y_AC6[:8], label=label6, color='#fee08b')

        DA50_16 =[15.   ,       1.31806858  ,1.30729771 , 1.1511028   ,0.    ,      2.17803512,
  0.08721391 ,-1.47060572 , 0.13654338]
        x_AC7, y_AC7 = points(DA50_16, AC.fuselage_width)
        label7 = '250 000 steps'
        plt.plot(x_AC7[:8], y_AC7[:8], label=label7, color='#abdda4')

        DA50_17 =  [15.      ,    1.25741154  ,1.25741154 , 1.25741154 , 0.15595992  ,2.41893998,
  0.12881251 ,-1.90039303  ,0.1352848 ]
        x_AC8, y_AC8 = points(DA50_17, AC.fuselage_width)
        label8 = '500 000 steps'
        plt.plot(x_AC8[:8], y_AC8[:8], label=label8, color='#3288bd')

  #       DA50_mil =  [ 1.50000000e+01 , 1.24822857e+00 , 1.24822857e+00  ,1.24822857e+00,
  # 0.00000000e+00,  2.16859982e+00 , 7.91447461e-03 ,-1.65078955e+00,
  # 1.36788545e-01]
  #       x_AC2, y_AC2 = points(DA50_mil, AC.fuselage_width)
  #       label2 = '1 000 000 steps'
  #       plt.plot(x_AC2[:8], y_AC2[:8], label=label2, color='#238b45')
  #
  #       DA50_best = [15.      ,    1.22769383 , 1.22769383 , 1.22769383 , 0.       ,   2.21354333,
  # 0.      ,   -1.61572937 , 0.14116904]
  #       x_AC10, y_AC10 = points(DA50_best, AC.fuselage_width)
  #       label10 = '1 272 000 steps'
  #       plt.plot(x_AC10[:8], y_AC10[:8], label=label10, color="#006d2c")



        DA50_15mil = [15.    ,      1.24575267 , 1.24575267 , 1.24427519 , 0.          ,2.35094735,
  0.      ,   -1.55802153 , 0.1374291 ]
        x_AC9, y_AC9 = points(DA50_15mil, AC.fuselage_width)
        label9 = '1 500 000 steps'
        plt.plot(x_AC9[:8], y_AC9[:8], label=label9, color='#01665e')


        plt.vlines(0, -2, 2.8, colors='#f7fbff',linestyles='dashed')
        plt.vlines(0, -2, 1, colors='#525252', linestyles='dashed')
        fig.gca().legend(loc="upper right")
        # fig.axes.yaxis.set_ticklabels([])
        plt.show()

    #### ANALYSIS 16: F50 planforms
    if ANALYSIS[15] == 1:
        # SR22 optimized
        # PSO optimized
        # Base max
        # TL
        # CL
        design_vector_start = [AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start,
                               AC.sweep_le_start, AC.yoffset_kink_start, AC.zoffset_tip_start,
                               AC.twist_start, AC.thickness_chord_start]

        # SORTING ML RESULTS

        # PLOTTING planforms results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        plt.axis('equal')
        # plt.title(f'Optimized F50 wing planforms')
        x_AC1, y_AC1 = points(design_vector_start, AC.fuselage_width)
        label1 = 'Starting planform'
        plt.plot(x_AC1[:8], y_AC1[:8], color='#969696', label=label1)
        plt.vlines(0, -2, 0.8, colors='#525252',linestyles='dashed')

        DA50_BASE = [36.    ,      2.41856448 , 2.41856448  ,2.41856448,  0.    ,      6.187495,
  0.      ,   -1.779763    ,0.14804499]
        x_AC2, y_AC2 = points(DA50_BASE, AC.fuselage_width)
        label2 = 'F50 model'
        plt.plot(x_AC2[:8], y_AC2[:8], label=label2, color ='#4292c6')

        DA50_TL = [36.      ,    2.84330467  ,2.84330467 , 1.69146407 , 0.     ,     5.48432093,
  0.      ,   -0.67992292 , 0.14532349]
        x_AC3, y_AC3 = points(DA50_TL, AC.fuselage_width)
        label3 = 'TL model'
        plt.plot(x_AC3[:8], y_AC3[:8], label=label3, color ="#807dba")

        DA50_CL = [33.76    ,    4.43333333 , 2.21666667 , 1.99666667  ,0.     ,     5.39266667,
  0.    ,     -1.2298794  , 0.14776667]
        x_AC4, y_AC4 = points(DA50_CL, AC.fuselage_width)
        label4 = 'CL model'
        plt.plot(x_AC4[:8], y_AC4[:8], label=label4, color ="#78c679")

        DA50_SR22 = [36.     ,     2.45757079 , 2.45757079 , 2.45757079  ,0.     ,     5.45837945,
  0.     ,    -2.89308759 , 0.14411952]
        x_AC5, y_AC5 = points(DA50_SR22, AC.fuselage_width)
        label5 = 'SR22 model'
        plt.plot(x_AC5[:8], y_AC5[:8], label=label5, color ="#737373")

        #####PSO PLOTTING
        number_PSO = 1
        aircraft_config = 'F50'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        sorted_CL_CD_PSO = sorted(CL_CD_PSO)
        best_CL_CD_PSO = sorted_CL_CD_PSO[-10:]
        best_CL_CD_PSO.reverse()
        best_idx_PSO = []
        best_design_PSO = []

        for idx in range(len(best_CL_CD_PSO)):
            best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

        for idx in range(len(best_CL_CD_PSO)):
            best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

        # PLOTTING ML results
        x_AC_PSO = []
        y_AC_PSO = []

        for design in range(number_PSO):
            x_ACi, y_ACi = points(best_design_PSO[design], AC.fuselage_width)
            x_AC_PSO.append(x_ACi)
            y_AC_PSO.append(y_ACi)
            # labeli = f'PSO design {design + 1}, CL/CD:{round(best_CL_CD_ML[design],3)}'
            plt.plot(x_AC_PSO[design][:8], y_AC_PSO[design][:8], color='#ef6548', label='PSO')

        plt.vlines(0, -6, 5, colors='#f7fbff',linestyles='dashed')
        plt.vlines(0, -6, 1, colors='#525252', linestyles='dashed')
        fig.gca().legend(loc="upper right")
        plt.show()


    ##### ANALYSIS 17: ASK21 planforms
    if ANALYSIS[16] == 1:
        # SR22 optimized
        # PSO optimized
        # Base max
        # TL
        # CL
        design_vector_start = [AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start,
                               AC.sweep_le_start, AC.yoffset_kink_start, AC.zoffset_tip_start,
                               AC.twist_start, AC.thickness_chord_start]

        # SORTING ML RESULTS

        # PLOTTING planforms results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        plt.axis('equal')
        plt.title(f'Optimized ASK21 wing planforms')
        x_AC1, y_AC1 = points(design_vector_start, AC.fuselage_width)
        label1 = 'starting wing shape'
        plt.plot(x_AC1, y_AC1, color='black', label=label1)

        DA50_BASE = [13.55954521 , 2.18754911  ,1.52      ,  0.40250973 , 0.22689404 , 2.17866667,
  0.57184081 , 1.71276132 , 0.1298    ]
        x_AC2, y_AC2 = points(DA50_BASE, AC.fuselage_width)
        label2 = 'ASK21 model'
        plt.plot(x_AC2, y_AC2, label=label2)

        DA50_TL = [16.93339393 , 1.50063514 , 1.01290751 , 0.48394011 , 0.26666667 , 5.05112673,
  0.58066667 , 0.07238035,  0.16146166]
        x_AC3, y_AC3 = points(DA50_TL, AC.fuselage_width)
        label3 = 'TL model'
        plt.plot(x_AC3, y_AC3, label=label3)

        DA50_CL = [15.72  ,      1.4802309 ,  1.12     ,   0.612  ,     0.   ,       4.472,
  0.66804896 , 0.8        , 0.1557217 ]
        x_AC4, y_AC4 = points(DA50_CL, AC.fuselage_width)
        label4 = 'CL model'
        plt.plot(x_AC4, y_AC4, label=label4)

        DA50_SR22 = [1.70686856e+01 ,1.49846316e+00, 1.01859784e+00 ,5.18666667e-01,
 0.00000000e+00 ,5.04533333e+00, 5.98255808e-01 ,4.50840890e-03,
 1.59800000e-01]
        x_AC5, y_AC5 = points(DA50_TL, AC.fuselage_width)
        label5 = 'SR22 model'
        plt.plot(x_AC5, y_AC5, label=label5)

        #####PSO PLOTTING
        number_PSO = 1
        aircraft_config = 'ASK21'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        sorted_CL_CD_PSO = sorted(CL_CD_PSO)
        best_CL_CD_PSO = sorted_CL_CD_PSO[-10:]
        best_CL_CD_PSO.reverse()
        best_idx_PSO = []
        best_design_PSO = []

        for idx in range(len(best_CL_CD_PSO)):
            best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

        for idx in range(len(best_CL_CD_PSO)):
            best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

        # PLOTTING ML results
        x_AC_PSO = []
        y_AC_PSO = []

        for design in range(number_PSO):
            x_ACi, y_ACi = points(best_design_PSO[design], AC.fuselage_width)
            x_AC_PSO.append(x_ACi)
            y_AC_PSO.append(y_ACi)
            # labeli = f'PSO design {design + 1}, CL/CD:{round(best_CL_CD_ML[design],3)}'
            plt.plot(x_AC_PSO[design], y_AC_PSO[design], color='blue', label='PSO')

        fig.gca().legend(loc="upper left")
        # fig.axes.yaxis.set_ticklabels([])
        plt.show()




    ##### new plotter
    if ANALYSIS[16] == 2:
        # SR22 optimized
        # PSO optimized
        # Base max
        # TL
        # CL
        design_vector_start = [AC.wing_span_start, AC.chord_root_start, AC.chord_kink_start, AC.chord_tip_start,
                               AC.sweep_le_start, AC.yoffset_kink_start, AC.zoffset_tip_start,
                               AC.twist_start, AC.thickness_chord_start]

        # SORTING ML RESULTS

        # PLOTTING planforms results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        plt.axis('equal')
        # plt.title(f'Optimized F50 wing planforms')
        x_AC1, y_AC1 = points(design_vector_start, AC.fuselage_width)
        label1 = 'Starting planform'
        plt.plot(x_AC1[:8], y_AC1[:8], color='#969696', label=label1)
        plt.vlines(0, -2, 0.8, colors='#525252', linestyles='dashed')

        DA50_BASE = [13.55954521 , 2.18754911  ,1.52      ,  0.40250973 , 0.22689404 , 2.17866667,  0.57184081 , 1.71276132 , 0.1298    ]
        x_AC2, y_AC2 = points(DA50_BASE, AC.fuselage_width)
        label2 = 'ASK21 model'
        plt.plot(x_AC2[:8], y_AC2[:8], label=label2, color='#4292c6')

        DA50_TL = [16.93339393 , 1.50063514 , 1.01290751 , 0.48394011 , 0.26666667 , 5.05112673,  0.58066667 , 0.07238035,  0.16146166]
        x_AC3, y_AC3 = points(DA50_TL, AC.fuselage_width)
        label3 = 'TL model'
        plt.plot(x_AC3[:8], y_AC3[:8], label=label3, color="#807dba")

        DA50_CL = [15.72  ,      1.4802309 ,  1.12     ,   0.612  ,     0.   ,       4.472,  0.66804896 , 0.8        , 0.1557217 ]
        x_AC4, y_AC4 = points(DA50_CL, AC.fuselage_width)
        label4 = 'CL model'
        plt.plot(x_AC4[:8], y_AC4[:8], label=label4, color="#78c679")

        DA50_SR22 = [1.70686856e+01 ,1.49846316e+00, 1.01859784e+00 ,5.18666667e-01, 0.00000000e+00 ,5.04533333e+00, 5.98255808e-01 ,4.50840890e-03, 1.59800000e-01]
        x_AC5, y_AC5 = points(DA50_SR22, AC.fuselage_width)
        label5 = 'SR22 model'
        plt.plot(x_AC5[:8], y_AC5[:8], label=label5, color="#737373")

        #####PSO PLOTTING
        number_PSO = 1
        aircraft_config = 'ASK21'
        directory_PSO = 'PSO_Results' + '/' + 'Results_' + aircraft_config
        results_PSO_list = os.listdir(directory_PSO)
        CL_CD_PSO = []
        design_vectors_PSO = []

        for f in range(len(results_PSO_list)):
            CL_CD = 0
            file_name = results_PSO_list[f]
            file_directory = directory_PSO + '/' + file_name

            with open(file_directory, 'r') as fp:
                lines = len(fp.readlines())
            if lines > 0:
                CL_CD = - np.loadtxt(file_directory, skiprows=106, usecols=11, max_rows=1)
                design_vector = []
                for dv in range(9):
                    dv_i = - np.loadtxt(file_directory, skiprows=106, usecols=2 + dv, max_rows=1)
                    design_vector.append(-dv_i)
            design_vectors_PSO.append(design_vector)
            CL_CD_PSO.append(CL_CD)

        sorted_CL_CD_PSO = sorted(CL_CD_PSO)
        best_CL_CD_PSO = sorted_CL_CD_PSO[-10:]
        best_CL_CD_PSO.reverse()
        best_idx_PSO = []
        best_design_PSO = []

        for idx in range(len(best_CL_CD_PSO)):
            best_idx_PSO.append(CL_CD_PSO.index(best_CL_CD_PSO[idx]))

        for idx in range(len(best_CL_CD_PSO)):
            best_design_PSO.append(design_vectors_PSO[best_idx_PSO[idx]])

        # PLOTTING ML results
        x_AC_PSO = []
        y_AC_PSO = []

        for design in range(number_PSO):
            x_ACi, y_ACi = points(best_design_PSO[design], AC.fuselage_width)
            x_AC_PSO.append(x_ACi)
            y_AC_PSO.append(y_ACi)
            # labeli = f'PSO design {design + 1}, CL/CD:{round(best_CL_CD_ML[design],3)}'
            plt.plot(x_AC_PSO[design][:8], y_AC_PSO[design][:8], color='#ef6548', label='PSO')

        plt.vlines(0, -3, 3, colors='#f7fbff', linestyles='dashed')
        plt.vlines(0, -3, 1, colors='#525252', linestyles='dashed')
        fig.gca().legend(loc="upper right")
        plt.show()

    ##### ANALYSIS 18: Average time to result

    if ANALYSIS[17] == 1:
        # res = Results('DA_50')
        # objectives = res.objectives
        objectives = [26.522295096782315, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.504088332324162, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.504089738370197, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.501570959873373, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.535828903927257, 26.51263574421521, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.508184388795136, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.506295890514536, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.507100093082535, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.507027320968877, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.50581878657168, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.51798535268845, 26.5, 26.5, 26.5, 26.5, 26.5, 26.522720683272066, 26.5, 26.50634791624006, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.508502681390517, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.501128288193385, 26.5, 26.5, 26.5, 26.5, 26.52060100351759, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.527290348474807, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.508613820635716, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.502403554116448, 26.5, 26.508974895326276, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.504559788879117, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.5, 26.517935874083197, 26.5, 26.503110047269473, 26.5, 26.5]
        time =  4.7958695478439335







        PSO_CLCD, PSO_design_vectors = get_PSO()
        PSO_average = sum(PSO_CLCD) / len(PSO_CLCD)

        higher = []
        average = sum(objectives) / len(objectives)

        for i in range(len(objectives)):
            if objectives[i] > average:
                higher.append(objectives.append)
        prob_pos = len(higher) / len(objectives)

        prob = 0.99
        n = 1

        prob_calc = 1 - (1-prob_pos)**n
        print(prob_calc)

        while prob_calc < prob:
            n += 1
            prob_calc = 1 - (1-prob_pos)**n
        print(prob_calc)
        print('time_ave', n * time)

# tensorboard plotting
    if ANALYSIS[18] == 1:
        import matplotlib.ticker as ticker
        import pandas as pd

        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)

        csv_file = "DA50_BASE.csv"
        df = pd.read_csv(csv_file)


        column_rewards = 'Value'
        rewards = df[column_rewards].tolist()
        column_steps = 'Step'
        steps = df[column_steps].tolist()


        csv_fileTL = "DA50_TL.csv"
        dfTL = pd.read_csv(csv_fileTL)

        column_rewardsTL = 'Value'
        rewardsTL = dfTL[column_rewards].tolist()
        column_stepsTL = 'Step'
        stepsTL = dfTL[column_steps].tolist()

        csv_fileCL = "DA50_CL.csv"
        dfCL = pd.read_csv(csv_fileCL)

        column_rewardsCL = 'Value'
        rewardsCL = dfCL[column_rewards].tolist()
        rewardsCL[:4] = [0,0,0,0]
        column_stepsCL = 'Step'
        stepsCL = dfCL[column_steps].tolist()
        stepsCL = [number - 1302528 for number in stepsCL]

        fig = plt.figure()
        ax = plt.subplot(111)

        plt.plot(steps, rewards, color='#4292c6', label='DA50 model')
        plt.plot(stepsTL, rewardsTL, color="#807dba", label='TL model')
        plt.plot(stepsCL, rewardsCL, color="#78c679", label='CL model')
        plt.xlabel('Training steps')
        plt.ylabel('Mean rewards')
        ax.grid('on', linestyle='dashed')
        fig.gca().legend(loc="lower right")
        formatter.set_powerlimits((-3, 4))  # Adjust the range of exponents if needed
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.show()

# Sigma result plot
#
#     if ANALYSIS[19] ==1:
